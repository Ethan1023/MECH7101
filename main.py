from loaddata import load
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import re
import pickle
import os
import logging
import scipy.optimize as opt
import os
import sys

matplotlib.use("WebAgg")
plt.rcParams["figure.figsize"] = (18, 8.5)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--manual', action='store_true', help='edit data ranges manually')
parser.add_argument('-c', '--manualcali', action='store_true', help='edit calibration data ranges manually')
parser.add_argument('-d', '--debug', action='store_true', help='print extra info')
parser.add_argument('-r', '--reload', action='store_true', help='reload raw data')
args = parser.parse_args()
if args.debug:
    logging.basicConfig(level=20)

RHO = 1.225

FOLDERS = ('data/T3', 'data/T3_repeat')
SAT_MAX = 1e5
SAT_MIN = -2e6
OUTLIER = 1e5  # copy values forwards when there is a jump larger than this
# Board 3
CALIBRATIONS = (-1.2764e6, -1.3224e6, -1.3051e6, -1.2491e6, -1.2329e6, -1.2511e6, -1.2753e6, -1.2463e6, -1.3100e6, -1.2395e6)
# Board 1
CALIBRATIONS = (-1.3308e6, -1.1619e6, -1.3071e6, -1.2534e6, -1.2971e6, -1.3484e6, -1.2510e6, -1.3656e6, -1.3035e6, -1.2218e6)
RPMS = (700, 600, 500, 400, 300, 700)
EXCLUDE = np.array([4, 9]) - 1
CALI_STEPS = ((8, 9), (8, 9))
CALI_WEIGHTS = ((list(range(10, 81, 10)),
                 list(range(10, 81, 10))+[0]),
                (list(range(10, 71, 10))+[0],
                 list(range(10, 81, 10))+[0]))
HOLE_I = np.array([ 1,   2,   3,   5,  6,   7,   8,  10])
HOLE_X = np.array([36, -65, -65, -36, 65,  36, -36,  65]) / 1000
HOLE_Y = np.array([65,  18, -18, -65, 18, -65,  65, -18]) / 1000
ROPE_DIR = np.array([0, -90, -90, 180, 90, 180, 0, 90]) * np.pi/180  # Assume the lined up well (they did)
ROPE_ANGLE = np.array([1124, 609]) - np.array([1031, 467])  # coords from a pic
ROPE_ANGLE = np.arctan(ROPE_ANGLE[1]/ROPE_ANGLE[0])

ROPE_ANGLES_1 = np.ones(8) * ROPE_ANGLE
ang_r = np.arctan(-(773-555) / (907-1021))
ang_l = np.arctan((725-529) / (1124-972))
ang_f = (ang_r + ang_l)/2
ang_b = (ang_r + ang_l)/2
ROPE_ANGLES_2 = np.array([ang_f, ang_l, ang_l, ang_b, ang_r, ang_b, ang_f, ang_r])

ROPE_ANGLES = np.array([ROPE_ANGLES_1, ROPE_ANGLES_2])

print(f'rope angles = {ROPE_ANGLES*180/np.pi}')

T_X = -np.sin(ROPE_DIR) * np.cos(ROPE_ANGLES)
T_Y = -np.cos(ROPE_DIR) * np.cos(ROPE_ANGLES)
T_Z = np.ones(len(ROPE_DIR)) * np.sin(ROPE_ANGLES)

dT_X_dth1 = -np.cos(ROPE_DIR) * np.cos(ROPE_ANGLES)
dT_Y_dth1 = np.sin(ROPE_DIR) * np.cos(ROPE_ANGLES)
dT_X_dth2 = np.sin(ROPE_DIR) * np.sin(ROPE_ANGLES)
dT_Y_dth2 = np.cos(ROPE_DIR) * np.sin(ROPE_ANGLES)

th1_uncertainty = 10 * np.pi/180
th2_uncertainty = 10 * np.pi/180
tension_uncertainty_percent = 0.5/100

# Other @ 20
RHO = 1.204   # Density (kg/m3)
MU = 1.825 * 10**(-5)

# Geometry
AREA_FRONT = 6784.4388 * 1000**-2   # Frontal projected area (m2)
AREA_SIDE = 5936.2951 * 1000**-2    # Side projected are (m2)
AREA_ANGLE = 8682.7960 * 1000**-2   # Projected area at 45 degress (m2)
L_GEOM = 101.9821 / 1000            # Width from front/rear (m)
W_GEOM = 99.4821 / 1000             # Width from sides (m)
H_GEOM = 86.6210 / 1000             # Height (m)
NAMES_AREAS = ['front', 'side', 'angle']
NAMES_GEOM = ['length', 'width', 'height']
AREAS = np.array([AREA_FRONT, AREA_SIDE, AREA_ANGLE])
GEOM = np.array([L_GEOM, W_GEOM, H_GEOM])
def area_angle(angle):
    return np.abs(np.cos(angle)*AREA_FRONT) + np.abs(np.sin(angle)*AREA_SIDE)

# Maximum friction values from calibration,
FRICTION_X = 190/1000
FRICTION_Y = 90/1000
SYMMETRICAL_FIT = True


if not os.path.exists('ropes.png') or True:
    lengthx = (130-90)/2000
    lengthy = (130-100)/2000
    for i, x, y, tx, ty in zip(HOLE_I, HOLE_X, HOLE_Y, T_X[0], T_Y[0]):
        #y2 = y - length * np.cos(d)
        #x2 = x - length * np.sin(d)
        plt.scatter(x, y)
        plt.plot([x, x+np.sign(round(tx,10))*lengthx], [y, y+np.sign(round(ty,10))*lengthy], label=f'Load cell {i}')
    plt.plot([-90/2000, -90/2000, 90/2000, 90/2000, -90/2000], [-100/2000, 100/2000, 100/2000, -100/2000, -100/2000], label='Castle base')
    plt.legend()
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    fname = 'ropes.png'
    plt.savefig(fname, dpi=200)
    if sys.platform == 'linux':
        os.system(f'convert {fname} -trim {fname}')
    plt.close()

def cw_func(x, p):
    ''' function for cw '''
    a, b, c, d = p
    Re, theta = x
    return a*(1 + b*np.sin(theta - c)) * Re**d

def cw_func_2(x, p):
    ''' function for cw '''
    #a, b, c, d = p
    Re, theta = x
    add = 1
    for i in range(0, int(len(p)/2-1)):
        add += p[i+1] * np.sin((i+1)*theta - p[i+2])
    add *= p[0] * Re**p[-1]
    return add

def cw_func_sym(x, p):
    ''' function for cw '''
    #a, b, c, d = p
    Re, theta = x
    if hasattr(Re, '__iter__'):
        add = np.ones(Re.shape)
    else:
        add = 1
    for i in range(1, len(p)-1):
        add += p[i] * np.cos((2**(i-1))*theta)
    add *= p[0] * Re**p[-1]
    return add

def print_cw_func_sym(p):
    string = ''
    if not p[0] == 1:
        string += f'{formnum(p[0])}'
    string += r'\left(1'
    if p[1] < 0:
        string += ' - '
    else:
        string += ' + '
    for i in range(1, len(p)-1):
        if i == 1:
            string += f'{formnum(abs(p[i]))}\cos('
        else:
            string += f'{formnum(abs(p[i]))}\cos({int(2**(i-1))}'
        string += r'\theta)'
        if i < len(p)-2:
            if p[i+1] >= 0:
                string += ' + '
            else:
                string += ' - '
    string += r'\right)\textrm{Re}^{'
    string += f'{formnum(p[-1])}'
    string += '}'
    return string

def calc_RMSE(p, func, data):  # Calc RMSE with paramteters
    y0 = data[0]
    x = data[1:]
    y1 = func(x, p)
    axis = 0
    if hasattr(p[0], '__iter__'):
        axis = 1
    mse = np.sqrt(np.mean((y1-y0)**2,axis=axis))  # Sum of squared errors
    return mse

def optimise(data, i=-1):
    #res = opt.differential_evolution(calc_RMSE, ((-1000, 1000), (-10, 10), (0, 2*np.pi), (-10, 10)), workers=32, args=(cw_func_2, data), updating='deferred')
    if i == -1:
        if not SYMMETRICAL_FIT:
            func = cw_func_2
            res = opt.differential_evolution(calc_RMSE, ((-1000, 1000), (-10, 10), (0, 2*np.pi), (-10, 10), (0, 2*np.pi), (-10, 10)), workers=32, args=(func, data), updating='deferred')
        else:
            func = cw_func_sym
            #res = opt.differential_evolution(calc_RMSE, ((-1000, 1000), (-10, 10), (-10, 10), (-10, 10)), workers=32, args=(func, data), updating='deferred')
            res = opt.differential_evolution(calc_RMSE, ((-10000, 10000), (-1, 1), (-1, 1), (-1, 1), (-10, 10)), workers=32, args=(func, data), updating='deferred')
        print(f'Coarse:')
        print(f'RMSE = {res.fun}')
        print(f'Parameters = {res.x}')
        res = opt.minimize(calc_RMSE, res.x, args=(func, data))
        print(f'Fine:')
        print(f'RMSE = {res.fun}')
        p = res.x
    elif i == 0:
        p = [ 7.07560841e+04, -1.27343348e-01, 2.04187165e-01, 2.93804988e-02, -1.02494130e+00]
    elif i == 1:
        p = [ 1.82738943e+05, 6.62006467e-02, -2.29919040e-01, 1.74996822e-02, -1.10384633e+00]
    elif i == 2:
        p = [ 9.08693945e+04, -1.16130819e-02, -8.15230261e-02, 4.01776247e-02, -1.04005935e+00]
        #p = [ 9.57538179e+03, -5.57550165e-03, -7.83558909e-02, 2.13923865e-02, -8.24532479e-01]
    print(f'Parameters = {p}')
    print(f'RMSE = {calc_RMSE(p, cw_func_sym, data)}')
    print(f'Function is: {print_cw_func_sym(p)}')
    return p

def formnum(num, sci=3, dec=3):
    string = ''
    exp = int(np.floor(np.log10(np.abs(num))))
    if abs(exp) > sci:
        num /= 10**exp
        string += f'{round(num,dec)}'
        string += r'\times10^{'
        string += f'{exp}'
        string += r'}'
    elif num < 1:
        string += f'{round(round(num*10**-exp,dec)*10**exp,10)}'
    else:
        string += f'{round(num,dec)}'
    return string

def main():
    '''
    '''

    all_the_data = []
    for i, folder in enumerate(FOLDERS):
        if args.reload or not os.path.exists(folder+'_data.csv'):
            print(f'Folder = {folder}')
            cali, data, pressure = load(folder)
            # 3D arrays: angle x sensor x rpm
            all_data, columns = prepare_all_data(folder, data, pressure)
            save_csv(folder+'_data.csv', columns, all_data)
            print('saved data from csv')
            cali_data, cali_columns = prepare_cali_data(folder, cali, CALI_WEIGHTS[i], CALI_STEPS[i])
            save_csv(folder+'_cali.csv', cali_columns, cali_data)
            print('saved calibration data from csv')
        else:
            all_data, columns = load_csv(folder+'_data.csv')
            print('loaded data from csv')
            cali_data, cali_columns = load_csv(folder+'_cali.csv')
            print('loaded calibration data from csv')
        cali_tensions = cali_data[:, 3:11] - cali_data[-1, 3:11]
        if True:
            sk = i
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colours = prop_cycle.by_key()['color']
            colour_index = 0
            fx = T_X[i] * cali_tensions
            fy = T_Y[i] * cali_tensions
            fz = T_Z[i] * cali_tensions
            Fx = np.sum(fx, axis=1)
            Fy = np.sum(fy, axis=1)
            Fz = np.sum(fz, axis=1)
            Lx = -np.sin(cali_data[:,1]) * cali_data[:,0] * 9.81
            Ly = -np.cos(cali_data[:,1]) * cali_data[:,0] * 9.81
            mux = (np.abs(Lx) - np.abs(Fx)) #/ Fz
            muy = (np.abs(Ly) - np.abs(Fy)) #/ Fz
            splitind = np.where(np.diff(cali_data[:,1]) != 0)[0][0] + 1
            indicies = np.array(range(len(Fx)))
            #plt.plot(indicies[:splitind], Fx[:splitind]*1000, label='Fx', color=colours[colour_index])
            plt.plot(indicies[splitind:-1], Fx[splitind:-1]*1000, label='Fx (mN)', color=colours[colour_index])
            colour_index += 1
            plt.plot(indicies[:splitind-sk], Fy[:splitind-sk]*1000, label='Fy (mN)', color=colours[colour_index])
            #plt.plot(indicies[splitind:-1], Fy[splitind:-1]*1000, color=colours[colour_index])
            colour_index += 1
            plt.plot(indicies[:splitind-sk], Fz[:splitind-sk]*1000, label='Fz (mN)', color=colours[colour_index], ls=':')
            plt.plot(indicies[splitind:-1], Fz[splitind:-1]*1000, color=colours[colour_index], ls=':')
            colour_index += 1
            #plt.plot(indicies[:splitind-sk], -cali_data[:,0][:splitind-sk]*9.81*1000, label='force', color=colours[colour_index])
            #plt.plot(indicies[splitind:-1], -cali_data[:,0][splitind:-1]*9.81*1000, color=colours[colour_index])
            colour_index += 1
            colour_index = 0
            #plt.plot(indicies[:splitind], Lx[:splitind]*1000, label='Fx applied', color=colours[colour_index])
            plt.plot(indicies[splitind:-1], Lx[splitind:-1]*1000, label='Fx applied (mN)', color=colours[colour_index], ls='--')
            colour_index += 1
            plt.plot(indicies[:splitind-sk], Ly[:splitind-sk]*1000, label='Fy applied (mN)', color=colours[colour_index], ls='--')
            #plt.plot(indicies[splitind:-1], Ly[splitind:-1]*1000, color=colours[colour_index])
            colour_index += 1
            colour_index = 0
            #plt.plot(indicies[:splitind], mux[:splitind]*10000, label='Mu x', color=colours[colour_index])
            plt.plot(indicies[splitind:-1], mux[splitind:-1]*10000, label='Mu x (mN/10)', color=colours[colour_index], ls='-.')
            colour_index += 1
            plt.plot(indicies[:splitind-sk], muy[:splitind-sk]*10000, label='Mu y (mN/10)', color=colours[colour_index], ls='-.')
            #plt.plot(indicies[splitind:-1], muy[splitind:-1]*10000, color=colours[colour_index])
            colour_index += 2
            plt.plot(indicies[:splitind-sk], cali_data[:,1][:splitind-sk]*180/np.pi*10, label='angle (deci-degrees)', color=colours[colour_index])
            plt.plot(indicies[splitind:-1], cali_data[:,1][splitind:-1]*180/np.pi*10, color=colours[colour_index])
            colour_index += 1
            plt.xlabel('Test')
            plt.ylabel('Force / Angle')
            plt.grid(visible=True, which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.legend()
            fname = f'Calibration_{i+1}.png'
            plt.savefig(fname, dpi=200)
            if sys.platform == 'linux':
                os.system(f'convert {fname} -trim {fname}')
            #plt.show()
            plt.close()
        zero(all_data)
        all_tensions = all_data[:, 4:12]

        if True:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colours = prop_cycle.by_key()['color']
            def plotchanges(x, y, changelist, label, ls=None, c_ind=0, dashes=None):
                if ls is None and dashes is None:
                    ls = '-'
                for i in range(len(changelist)):
                    if i == 0:
                        if dashes:
                            plt.plot(x[0:changelist[i]], y[0:changelist[i]], label=label, dashes=dashes, color=colours[c_ind])
                        else:
                            plt.plot(x[0:changelist[i]], y[0:changelist[i]], label=label, ls=ls, color=colours[c_ind])
                    elif changelist[i] == 0:
                        if dashes:
                            plt.plot(x[changelist[i-1]:], y[changelist[i-1]:], dashes=dashes, color=colours[c_ind])
                        else:
                            plt.plot(x[changelist[i-1]:], y[changelist[i-1]:], ls=ls, color=colours[c_ind])
                    else:
                        if dashes:
                            plt.plot(x[changelist[i-1]:changelist[i]], y[changelist[i-1]:changelist[i]], dashes=dashes, color=colours[c_ind])
                        else:
                            plt.plot(x[changelist[i-1]:changelist[i]], y[changelist[i-1]:changelist[i]], ls=ls, color=colours[c_ind])

            thetas = all_data[:,2]      # Flow angles
            # Identify different angles
            changes = np.concatenate([np.where(np.diff(thetas) != 0)[0]+1, [0]])
            indicies = np.array(range(len(thetas))) + 1
            fx = T_X[i] * all_tensions  # Components of line tensions
            Fx_uncertainty = np.sqrt((np.sum(dT_X_dth1[i]*all_tensions*th1_uncertainty,axis=1))**2 + (np.sum(dT_X_dth2[i]*all_tensions*th2_uncertainty,axis=1))**2 + (np.sum(T_X[i]*all_tensions*tension_uncertainty_percent,axis=1))**2)
            fy = T_Y[i] * all_tensions
            Fy_uncertainty = np.sqrt((np.sum(dT_Y_dth1[i]*all_tensions*th1_uncertainty,axis=1))**2 + (np.sum(dT_Y_dth2[i]*all_tensions*th2_uncertainty,axis=1))**2 + (np.sum(T_Y[i]*all_tensions*tension_uncertainty_percent,axis=1))**2)
            fz = T_Z[i] * all_tensions
            Fx = np.sum(fx, axis=1)     # Total force in x direction
            Fy = np.sum(fy, axis=1)
            Fz = np.sum(fz, axis=1)
            #print(Fy_uncertainty/Fy)
            # Friction components
            Fx_friction = (np.abs(FRICTION_X*np.sin(thetas))) * np.sign(Fx)
            Fy_friction = (np.abs(FRICTION_Y*np.cos(thetas))) * np.sign(Fy)
            # Total loads
            Fx_total = Fx + Fx_friction
            Fy_total = Fy + Fy_friction
            
            # Rotated loads
            F_parallel = np.cos(thetas) * Fy + np.sin(thetas) * Fx
            F_normal = -np.sin(thetas) * Fy + np.cos(thetas) * Fx
            F_parallel_uncertainty = np.sqrt((np.sin(thetas)*Fy_uncertainty)**2 + (np.cos(thetas)*Fx_uncertainty)**2)
            F_normal_uncertainty = np.sqrt((np.cos(thetas)*Fy_uncertainty)**2 + (np.sin(thetas)*Fx_uncertainty)**2)
            F_parallel_total = np.cos(thetas) * Fy_total + np.sin(thetas) * Fx_total
            F_normal_total = -np.sin(thetas) * Fy_total + np.cos(thetas) * Fx_total

            # Projected area
            areas = area_angle(all_data[:,2])
            # Wind speeds
            vel = np.sqrt(2/RHO * -all_data[:,1])
            # Total loads
            #Fs = np.sqrt(Fx_total**2 + Fy_total**2)
            Fs = np.sqrt(Fx**2 + Fy**2)
            # Wind coefficients
            Cws = Fs / (RHO/2 * vel**2 * areas)
            # Reynold's number
            Re = RHO * vel * (L_GEOM*W_GEOM*H_GEOM)**(1/3) / MU
            # Arrange data
            data = np.array([Cws, Re, thetas])
            # Delete zero velocity elements
            data = np.delete(data, np.where(vel == 0)[0], axis=1)
            all_the_data.append(data)
            # Least squares fit
            p = optimise(data, i=2)
            # Predicted wind coefficients
            if SYMMETRICAL_FIT:
                Cw2 = cw_func_sym((Re, thetas), p)
            else:
                Cw2 = cw_func_2((Re, thetas), p)
            
            #where = np.where(all_data[:,2] == -np.pi/2)[0]
            #plt.plot(np.take(vel,where), np.take((RHO/2 * vel**2*areas),where), label='neg')
            #plt.plot(np.take(vel,where), np.take(F_parallel,where), label='negF')
            #where = np.where(all_data[:,2] == np.pi/2)[0]
            #plt.plot(np.take(vel,where), np.take((RHO/2 * vel**2*areas),where), label='pos')
            #plt.plot(np.take(vel,where), np.take(F_parallel,where), label='posF')
            #plt.legend()
            #plt.show()
            #plt.close()

            # net forces in x and y
            fname = f'Experiment_{i+1}.png'
            if not os.path.exists(fname) or True:
                colour_index = 0
                plotchanges(indicies, Fx*1000, label='Fx measured (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, Fx_total*1000, label='Fx wind load (mN)', changelist=changes, c_ind=colour_index, ls='-.')
                colour_index += 1
                plotchanges(indicies, Fy*1000, label='Fy measured (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, Fy_total*1000, label='Fy wind load (mN)', changelist=changes, c_ind=colour_index, ls='-.')
                colour_index += 1
                plotchanges(indicies, Fz*1000, label='Fz (mN)', changelist=changes, c_ind=colour_index)
                colour_index += 1
                #plotchanges(indicies, F_parallel*1000, label='F parallel (mN)', changelist=changes, c_ind=colour_index, ls=':')
                #colour_index += 1
                #plotchanges(indicies, F_normal*1000, label='F normal (mN)', changelist=changes, c_ind=colour_index, ls=':')
                #colour_index += 1
                #plt.plot(all_data[:,0], label='rpm')
                #plt.plot(-all_data[:,1]*10, label='dp')
                plotchanges(indicies, vel*10, label='Wind speed (deci-m/s)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plotchanges(indicies, thetas*180/np.pi, label='Angle (degrees)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plt.xlabel('Test')
                #plt.ylabel('Net force (mN) / rpm / deci-degrees / decipascal / cm/s')
                plt.ylabel('Net force / Angle / Wind speed')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.legend()
                plt.savefig(fname, dpi=200)
                if sys.platform == 'linux':
                    os.system(f'convert {fname} -trim {fname}')
                #plt.show()
                plt.close()

            # net forces parallel and normal
            fname = f'Experiment_{i+1}_aligned.png'
            if not os.path.exists(fname) or True:
                colour_index = 0
                plotchanges(indicies, F_parallel*1000, label='F measured parallel (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, F_parallel_total*1000, label='F wind load parallel (mN)', changelist=changes, c_ind=colour_index, ls='-.')
                colour_index += 1
                plotchanges(indicies, F_normal*1000, label='F measured normal (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, F_normal_total*1000, label='F wind load normal (mN)', changelist=changes, c_ind=colour_index, ls='-.')
                colour_index += 1
                #plotchanges(indicies, Fz*1000, label='Fz (mN)', changelist=changes, c_ind=colour_index)
                colour_index += 1
                #plotchanges(indicies, F_parallel*1000, label='F parallel (mN)', changelist=changes, c_ind=colour_index, ls=':')
                #colour_index += 1
                #plotchanges(indicies, F_normal*1000, label='F normal (mN)', changelist=changes, c_ind=colour_index, ls=':')
                #colour_index += 1
                #plt.plot(all_data[:,0], label='rpm')
                #plt.plot(-all_data[:,1]*10, label='dp')
                plotchanges(indicies, vel*10, label='Wind speed (deci-m/s)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plotchanges(indicies, thetas*180/np.pi*1, label='Angle (degrees)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plt.xlabel('Test')
                #plt.ylabel('Net force (mN) / rpm / deci-degrees / decipascal / cm/s')
                plt.ylabel('Net force / Angle / Wind speed')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.legend()
                plt.savefig(fname, dpi=200)
                if sys.platform == 'linux':
                    os.system(f'convert {fname} -trim {fname}')
                #plt.show()
                plt.close()

            fname = f'Experiment_{i+1}_uncertainty.png'
            # net forces parallel and normal with uncertainties
            if not os.path.exists(fname) or True:
                colour_index = 0
                plotchanges(indicies, F_parallel*1000, label='F measured parallel (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, F_parallel_uncertainty*1000, label='F uncertainty parallel (mN)', changelist=changes, c_ind=colour_index, dashes=[3,3,2,6])
                print(F_parallel_uncertainty)
                colour_index += 1
                plotchanges(indicies, F_normal*1000, label='F measured normal (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, F_normal_uncertainty*1000, label='F uncertainty normal (mN)', changelist=changes, c_ind=colour_index, dashes=[2,6,3,3])
                colour_index += 1
                #plotchanges(indicies, Fz*1000, label='Fz (mN)', changelist=changes, c_ind=colour_index)
                colour_index += 1
                #plotchanges(indicies, F_parallel*1000, label='F parallel (mN)', changelist=changes, c_ind=colour_index, ls=':')
                #colour_index += 1
                #plotchanges(indicies, F_normal*1000, label='F normal (mN)', changelist=changes, c_ind=colour_index, ls=':')
                #colour_index += 1
                #plt.plot(all_data[:,0], label='rpm')
                #plt.plot(-all_data[:,1]*10, label='dp')
                plotchanges(indicies, vel*10, label='Wind speed (deci-m/s)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plotchanges(indicies, thetas*180/np.pi*1, label='Angle (degrees)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plt.xlabel('Test')
                #plt.ylabel('Net force (mN) / rpm / deci-degrees / decipascal / cm/s')
                plt.ylabel('Net force / Angle / Wind speed')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.legend()
                plt.savefig(fname, dpi=200)
                if sys.platform == 'linux':
                    os.system(f'convert {fname} -trim {fname}')
                #plt.show()
                plt.close()

            # wind coefficient predictions
            fname = f'Experiment_{i+1}_cw_pre.png'
            if not os.path.exists(fname) or True:
                colour_index = 0
                plotchanges(indicies, Fs*10, label='F (deciN)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plt.plot(Cws, label='Cw', color=colours[colour_index])
                plt.plot(Cw2, label='Cw predicted', color=colours[colour_index], ls='-.')
                colour_index += 1
                plotchanges(indicies, Re/10000, changelist=changes, label='Re*10e-4', c_ind=colour_index)
                colour_index += 1
                plotchanges(indicies, vel, changelist=changes, label='Wind speed (m/s)', c_ind=colour_index, ls=':')
                colour_index += 1
                plotchanges(indicies, thetas*180/np.pi/100, changelist=changes, label='Angle (hecto-degrees)', c_ind=colour_index)
                plt.xlabel('Test')
                #plt.ylabel('Net force (mN) / rpm / deci-degrees / decipascal / cm/s')
                plt.ylabel('Wind coefficient / Reynolds / Angle / Net force / Wind speed')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.legend()
                plt.savefig(fname, dpi=200)
                if sys.platform == 'linux':
                    os.system(f'convert {fname} -trim {fname}')
                #plt.show()
                plt.close()

            if True:  # 3d contour of wind coefficient
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter(vel, thetas*180/np.pi, Cws)
                x_re = np.linspace(20000, 90000)
                y_ang = np.linspace(-np.pi, np.pi)
                X_re, Y_ang = np.meshgrid(x_re, y_ang)
                if SYMMETRICAL_FIT:
                    Z_cw = cw_func_sym((X_re, Y_ang), p)
                else:
                    Z_cw = cw_func_2((X_re, Y_ang), p)
                #ax.plot_surface(X_re, Y_ang, Z_cw)
                ax.set_xlabel('Wind speed (m/s)')
                ax.set_ylabel('Wind angle (degrees)')
                ax.set_zlabel('Wind coefficient')
                plt.show()
                plt.close()

            if True:  # Max and mean loads against test runs
                colour_index = 0
                plotchanges(indicies, Fs*1000, label='Total load on tethers (mN)', changelist=changes, c_ind=colour_index)
                colour_index += 1
                plotchanges(indicies, 1000*np.max(all_tensions, axis=1), label='Highest tether load (mN)', changelist=changes, c_ind=colour_index)
                plotchanges(indicies, 1000*np.mean(all_tensions, axis=1), label='Average tether load (mN)', changelist=changes, c_ind=colour_index, ls='-.')
                colour_index += 1
                plotchanges(indicies, vel*10, label='Wind speed (deci-m/s)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plotchanges(indicies, thetas*180/np.pi*1, label='Angle (degrees)', changelist=changes, c_ind=colour_index, ls=':')
                colour_index += 1
                plt.xlabel('Test')
                #plt.ylabel('Net force (mN) / rpm / deci-degrees / decipascal / cm/s')
                plt.ylabel('Force / Angle / Wind speed')
                plt.grid(visible=True, which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.legend()
                fname = f'Experiment_{i+1}_tether.png'
                plt.savefig(fname, dpi=200)
                if sys.platform == 'linux':
                    os.system(f'convert {fname} -trim {fname}')
                #plt.show()
                plt.close()
    # Combine data from both experiments
    all_the_data_np = np.zeros((len(all_the_data[0]), len(all_the_data[0][0])+len(all_the_data[1][0])))
    all_the_data_np[:,:len(all_the_data[0][0])] = all_the_data[0]
    all_the_data_np[:,len(all_the_data[0][0]):] = all_the_data[1]
    # fit
    #p = optimise(all_the_data_np, i=-1)
    p = optimise(all_the_data_np, i=2)
    Cw, Re, thetas = all_the_data_np
    # Predicted wind coefficients
    if SYMMETRICAL_FIT:
        Cw2 = cw_func_sym((Re, thetas), p)
    else:
        Cw2 = cw_func_2((Re, thetas), p)

    res = 200
    x_re = np.linspace(15000, 90000, res)
    y_ang = np.linspace(-np.pi, np.pi, res)
    X_re, Y_ang = np.meshgrid(x_re, y_ang)
    if SYMMETRICAL_FIT:
        Z_cw = cw_func_sym((X_re, Y_ang), p)
    else:
        Z_cw = cw_func_2((X_re, Y_ang), p)
    cs = plt.contourf(X_re, Y_ang*180/np.pi, Z_cw)
    plt.scatter(Re, thetas*180/np.pi, label='Data points')
    plt.legend()
    plt.xlabel('Reynolds number')
    plt.ylabel('Angle (degrees)')
    plt.colorbar(cs, label='Wind coefficient')
    #plt.show()
    fname = 'Empirical_Cw.png'
    plt.savefig(fname, dpi=200)
    if sys.platform == 'linux':
        os.system(f'convert {fname} -trim {fname}')
    plt.close()

def zero(all_data):
    zero_row = None
    for row in all_data:
        if row[0] == 0:
            zero_row = row.copy()
            row[12:20] -= zero_row[12:20]
            row[4:12] -= zero_row[4:12]
        elif zero_row is not None:
            row[12:20] += zero_row[12:20]
            row[4:12] -= zero_row[4:12]
        else:
            print(f'WARNING - Zero row not found')

def prepare_cali_data(folder, all_cali, cali_weights, cali_steps = (9, 9), baselog=25):
    remove_list = [None, None]
    if os.path.exists(folder + '/cali_mask.pkl'):
        with open(folder + '/cali_mask.pkl', 'rb') as file:
            remove_list = pickle.load(file)
    all_data = []
    ANGLE = (0, np.pi/2)
    for i, (cali, weights) in enumerate(zip(all_cali, cali_weights)):
        cali2 = cali
        for index in EXCLUDE:
            cali2 = cali2.drop(f'Strain {index+1}', axis=1, inplace=False)
        remove_list[i] = detect_steps(cali2, remove=remove_list[i], manual=args.manualcali, num=cali_steps[i], numcolumn=10-len(EXCLUDE), name='cali')
        with open(folder + '/cali_mask.pkl', 'wb') as file:
            pickle.dump(remove_list, file)
        # Find where steps occur (changes from remove to not remove)
        steps = np.diff(np.r_[remove_list[i],[1]])
        start_ind = np.where(steps < -0.5)[0]+1
        stop_ind = np.where(steps > 0.5)[0]+1
        logging.log(baselog-2, f'start_ind = {start_ind}')
        logging.log(baselog-2, f'stop_ind = {stop_ind}')
        if len(start_ind) > cali_steps[i]:
            logging.log(baselog, f'removing last start_ind of {start_ind[cali_steps[i]:]}')
            start_ind = start_ind[:cali_steps[i]]
        if len(stop_ind) > cali_steps[i]:
            logging.log(baselog, f'removing last stop_ind of {stop_ind[cali_steps[i]:]}')
            stop_ind = stop_ind[:cali_steps[i]]
        all_forces, all_stds = process(cali, start_ind, stop_ind, EXCLUDE)
        all_samples = stop_ind - start_ind
        for j, (samples, forces, stds) in enumerate(zip(all_samples, all_forces.T, all_stds.T)):
            all_data.append(np.r_[weights[j]/1000, ANGLE[i], samples, forces, stds])
    all_data = np.array(all_data)
    included = []
    for i in range(10):
        if i not in EXCLUDE:
            included.append(i)
    columns = ['weight (kg)', 'angle (rad)', 'samples'] + [f'force_{i+1} (N)' for i in included] + [f'std_{i+1} (N)' for i in included]
    return (all_data, columns)

def prepare_all_data(folder, data, pressure):
    '''
    Return data as single 2D array, along with column labels
    '''
    meas_forces, meas_stds, meas_samples, angles = average_all_data(data, folder)
    allruns = np.zeros((len(data)*(len(RPMS)+1), 4))
    allforces = np.zeros((len(allruns), 10-len(EXCLUDE)))
    allstds = np.zeros((len(allruns), 10-len(EXCLUDE)))
    newcolumns = ['RPM'] + [str(int(ang)) for ang in angles*180/np.pi]  # real dodgy
    pressure = pressure[newcolumns]
    index = 0
    pressure_numpy = pressure.to_numpy()
    pressure_numpy = pressure_numpy[1:, 1:] - pressure_numpy[0, 1:]
    for i, angle in enumerate(angles):
        for j, rpm in enumerate([0] + list(RPMS)):
            allruns[index,0] = rpm
            if j == 0:
                allruns[index,1] = 0
            else:
                allruns[index,1] = pressure_numpy[j-1, i]
            assert angle*180/np.pi == float(pressure.columns[i+1]), "Angles don't match!"
            allruns[index,2] = angle
            allruns[index,3] = meas_samples[i,j]
            allforces[index] = meas_forces[i,:,j]
            allstds[index] = meas_stds[i,:,j]
            index += 1
    all_data = np.c_[allruns, allforces, allstds]
    included = []
    for i in range(10):
        if i not in EXCLUDE:
            included.append(i)
    columns = ['rpm', 'pressure (Pa)', 'angle (rad)', 'samples'] + [f'force_{i+1} (N)' for i in included] + [f'std_{i+1} (N)' for i in included]
    return all_data, columns

def save_csv(name, columns, data):
    with open(name, 'w') as file:
        file.write(', '.join(columns) + '\n')
        for row in data:
            str_row = ', '.join(str(elm) for elm in row) + '\n'
            file.write(str_row)

def load_csv(name):
    columns = None
    data = []
    with open(name, 'r') as file:
        for row in file.readlines():
            if columns is None:
                columns = row.replace('\n', '').split(', ')
            else:
                data.append([float(elm) for elm in row.replace('\n', '').split(', ')])
    return np.array(data), columns

def average_all_data(data, folder, baselog=10):
    rpms = 7
    meas_forces = np.zeros((len(data), 10-len(EXCLUDE), rpms))
    meas_stds = np.zeros((len(data), 10-len(EXCLUDE), rpms))
    meas_samples = np.zeros((len(data), rpms))
    angles = np.zeros(len(data))
    remove_list = [None for dat in data]
    if os.path.exists(folder + '/mask.pkl'):
        with open(folder + '/mask.pkl', 'rb') as file:
            remove_list = pickle.load(file)
    for i, (angle, df) in enumerate(data):
        logging.log(baselog-1, f'{i+1}/{len(data)}')
        angles[i] = angle
        sanitise_forwards(df)
        remove_list[i] = detect_steps(df, remove=remove_list[i], manual=args.manual)
        # Find where steps occur (changes from remove to not remove)
        steps = np.diff(np.r_[remove_list[i],[1]])
        start_ind = np.where(steps < -0.5)[0]+1
        stop_ind = np.where(steps > 0.5)[0]+1
        logging.log(baselog-2, f'start_ind = {start_ind}')
        logging.log(baselog-2, f'stop_ind = {stop_ind}')
        if len(start_ind) > 6:
            logging.log(baselog, f'removing last start_ind of {start_ind[6:]}')
            start_ind = start_ind[:6]
        if len(stop_ind) > 6:
            logging.log(baselog, f'removing last stop_ind of {stop_ind[6:]}')
            stop_ind = stop_ind[:6]
        first_steps = 20
        rows = df.loc[:first_steps].to_numpy()
        meas_forces[i,:,0] = np.delete(np.mean(rows[:,1:], axis=0) / CALIBRATIONS, EXCLUDE, axis=0)
        meas_stds[i,:,0] = np.delete(np.std(rows[:,1:]/CALIBRATIONS, axis=0), EXCLUDE, axis=0)
        meas_samples[i,0] = first_steps

        meas_forces[i,:,1:], meas_stds[i,:,1:] = process(df, start_ind, stop_ind, EXCLUDE)
        meas_samples[i,1:] = stop_ind - start_ind
        with open(folder + '/mask.pkl', 'wb') as file:
            pickle.dump(remove_list, file)
    return meas_forces, meas_stds, meas_samples, angles

def process(df, start_ind, stop_ind, exclude=None):
    meas_force = np.zeros((10, len(start_ind)))
    meas_std = np.zeros((10, len(start_ind)))
    for i, (start, stop) in enumerate(zip(start_ind, stop_ind)):
        rows = df.loc[start:stop].to_numpy()
        meas_force[:,i] = np.mean(rows[:,1:], axis=0) / CALIBRATIONS
        meas_std[:,i] = np.std(rows[:,1:]/CALIBRATIONS, axis=0)
    if exclude is not None:
        meas_force = np.delete(meas_force, exclude, axis=0)
        meas_std = np.delete(meas_std, exclude, axis=0)
    return meas_force, meas_std

def remove_steps(df, remove):
    return df.drop(df[remove==1].index)

def sanitise_forwards(df):
    for col in df.columns:
        values = df[col].values
        diff = np.r_[0, np.abs(np.diff(values))]
        while np.max(diff) > OUTLIER:
            values = values * (diff < OUTLIER) + np.r_[0, values[:-1]] * (diff >= OUTLIER)
            diff = np.r_[0, np.abs(np.diff(values))]
        df[col] = values

def detect_steps(df, manual=True, remove=None, num=6, numcolumn=10, name=None):
    if name is None:
        name = ''
    average_time = 10
    data = df.iloc[:,1:].to_numpy()
    if remove is None:
        step_time = np.mean(np.diff(df.iloc[:,0]))
        average_steps = int(round(average_time/step_time))
        step_threshold = 5e4
        cumsum = np.cumsum(data, axis=0)
        avg = (cumsum[average_steps:] - cumsum[:-average_steps]) / average_steps
        stepping_start = np.abs(data[average_steps:]-avg) > step_threshold
        stepping_end = np.abs(data[:-average_steps]-avg) > step_threshold
        stepping_end = np.r_[stepping_end[average_steps:], np.zeros((average_steps,numcolumn))]
        stepping = np.logical_or(stepping_start, stepping_end)
        remove = np.r_[np.ones(average_steps), np.any(stepping, axis=1)]
        get_6long(remove, num)
    user = ''
    while not user == 'done':
        plt.plot(remove*-1e6)
        for i in range(numcolumn):
            plt.plot(data.T[i])
        plt.savefig(f'identification/{name}{len(remove)}.png')
        if manual:
            plt.show()
        plt.close()
        if manual:
            user = input("Enter range to remove or enter 'done': ")
            if user == 'fill':
                get_6long(remove, num)
            elif user == 'regen':
                detect_steps(df, manual=True, remove=None)
            elif user == 'clear':
                remove = np.zeros(remove.shape)
            else:
                matches = re.search(r'([0-9]+)[:](-?[0-9]+)', user)
                if matches is not None:
                    start = int(matches.group(1))
                    stop = int(matches.group(2))
                    remove[start:stop] = True
        else:
            user = 'done'
    return remove

def get_6long(remove, num=6):
    longest_6 = [0 for _ in range(num)]
    start = None
    for i in range(len(remove)):
        if remove[i] == 0 and start is None:
            start = i
        if (remove[i] == 1 or i == len(remove)-1) and start is not None:
            diff = i - start
            index = longest_6.index(min(longest_6))
            if diff > longest_6[index]:
                longest_6[index] = diff
            start = None
    print(longest_6)
    if min(longest_6) < 200:
        print(f'WARNING - step likely missed!')
    start = None
    for i in range(len(remove)):
        if remove[i] == 0 and start is None:
            start = i
        if (remove[i] == 1 or i == len(remove)-1) and start is not None:
            diff = i - start
            if diff < min(longest_6):
                remove[start:i] += True
            start = None

def sanitise_midpoint(df):
    for col in df.columns:
        colt = df[col].values
        diffcol = np.abs(np.diff(colt))
        mindiffcol = np.concatenate([[0], np.minimum(diffcol[1:], diffcol[:-1]), [0]])
        avgcol = np.concatenate([[colt[0]],(colt[2:] + colt[:-2])/2, [colt[-1]]])
        newvalues = colt * (mindiffcol < OUTLIER) + avgcol * (mindiffcol >= OUTLIER)
        index = np.where(newvalues == min(newvalues))[0][0]
        df[col] = newvalues
        df[col] = np.maximum(df[col], np.ones(df[col].shape)*SAT_MIN)
        df[col] = np.minimum(df[col], np.ones(df[col].shape)*SAT_MAX)
        #print(df[col])

def plot(df):
    col0 = None
    for col in df.columns:
        if col0 is None:
            col0 = col
        else:
            plt.plot(df[col0], df[col], label=col)
    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
