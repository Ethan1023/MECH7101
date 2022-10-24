#!/usr/bin/python3
import os
import re
import numpy as np
import pandas as pd

def load(folder):
    cali0 = None
    cali90 = None
    tests = []
    pressure = None
    for filename in os.listdir(folder):
        file = os.path.join(folder, filename)
        if os.path.isfile(file):
            matches = re.match('(-?[0-9]+)deg', filename)
            if matches is not None:
                angle = float(matches.group(1)) * np.pi/180
                df = pd.read_csv(file,  skiprows=1)
                timeconvert = [3600, 60, 1, 0.01]
                times = np.array([sum([multi * float(num) for multi, num in zip(timeconvert, element.replace(',',':').split(':'))]) for element in df.iloc[:,0]])
                times -= times[0]
                df.iloc[:,0] = times
                if re.search('cali', filename) is not None:
                    if np.isclose(np.rad2deg(angle), 0):
                        cali0 = df
                    elif np.isclose(np.rad2deg(angle), 90):
                        cali90 = df
                    else:
                        assert 1 == 0, f'Unexpected Calibration at {angle*180/np.pi}'
                else:
                    toadd = True
                    for i in range(len(tests)):
                        if angle < tests[i][0] and toadd:
                            tests.insert(i, (angle, df))
                            toadd = False
                    if toadd:
                        tests.append((angle, df))
            elif re.match('pressure', filename):
                df = pd.read_csv(file)
                # Propogate zero rpm pressure readings
                cond = df.loc[0].isnull()
                columns = list(df.iteritems())
                for i, column in enumerate(columns):
                    if cond[i]:
                        name, vals = column
                        df.at[0, name] = columns[i-1][1][0]
                df.loc[0, cond] = df.loc[0, cond].replace('', None)
                pressure = df
    return (cali0, cali90), tests, pressure

def main():
    cali, data, pressure = load('data/T3')
    print(cali)
    print([d[0] for d in data])
    cali, data, pressure = load('data/T3_repeat')
    print(cali)
    print([d[0] for d in data])

if __name__ == '__main__':
    main()
