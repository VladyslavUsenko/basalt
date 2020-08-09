#!/usr/bin/env python3

import os
import sys
import json


datasets = ['Seq.', 'dataset-corridor1_512_16', 'dataset-magistrale1_512_16', 'dataset-room1_512_16', 'dataset-slides1_512_16']

# Other results.


vio = {
'ate' : ['VIO RMS ATE [m]'],
'time' : ['VIO Time [s]'],
'num_frames' : ['VIO Num. Frames']
}

out_dir = sys.argv[1]

def load_data(x, prefix, key):
    fname = out_dir + '/' + prefix + '_' + key
    if os.path.isfile(fname): 
        with open(fname, 'r') as f:
            j = json.load(f)
            res = round(j['rms_ate'], 3)
            x['ate'].append(float(res))
            x['time'].append(round(j['exec_time_ns']*1e-9, 3))
            x['num_frames'].append(j['num_frames'])
    else:
        x['ate'].append(float('Inf'))
        x['time'].append(float('Inf'))
        x['num_frames'].append(float('Inf'))


for key in datasets[1:]:
    load_data(vio, 'vio', key)


row_format ="{:>17}" + "{:>13}" * (len(datasets)-1)

datasets_short = [x[8:].split('_')[0] for x in datasets]

print('\nVisual-Inertial Odometry')
print(row_format.format(*datasets_short))

print(row_format.format(*vio['ate']))
#print(row_format.format(*vio['time']))
print(row_format.format(*vio['num_frames']))




