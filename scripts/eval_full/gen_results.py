#!/usr/bin/env python3

import os
import sys
import json


datasets = ['Seq.', 'MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult',
'MH_05_difficult', 'V1_01_easy', 'V1_02_medium',
'V1_03_difficult', 'V2_01_easy', 'V2_02_medium']


# Other results.


vio = {
'ate' : ['VIO RMS ATE [m]'],
'time' : ['VIO Time [s]'],
'num_frames' : ['VIO Num. Frames']
}

mapping = {
'ate' : ['MAP RMS ATE [m]'],
'time' : ['MAP Time [s]'],
'num_frames' : ['MAP Num. KFs']
}

pose_graph = {
'ate' : ['PG RMS ATE [m]'],
'time' : ['PG Time [s]'],
'num_frames' : ['PG Num. KFs']
}

pure_ba = {
'ate' : ['PG RMS ATE [m]'],
'time' : ['PG Time [s]'],
'num_frames' : ['PG Num. KFs']
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
    load_data(mapping, 'mapper', key)
    load_data(pose_graph, 'mapper_no_weights', key)
    load_data(pure_ba, 'mapper_no_factors', key)


row_format ="{:>17}" + "{:>13}" * (len(datasets)-1)

datasets_short = [x[:5] for x in datasets]

print('\nVisual-Inertial Odometry')
print(row_format.format(*datasets_short))

print(row_format.format(*vio['ate']))
#print(row_format.format(*vio['time']))
print(row_format.format(*vio['num_frames']))

print('\nVisual-Inertial Mapping')
print(row_format.format(*datasets_short))

print(row_format.format(*mapping['ate']))
#print(row_format.format(*mapping['time']))
print(row_format.format(*mapping['num_frames']))


print('\nPose-Graph optimization (Identity weights for all factors)')
print(row_format.format(*datasets_short))

print(row_format.format(*pose_graph['ate']))
#print(row_format.format(*pose_graph['time']))
print(row_format.format(*pose_graph['num_frames']))


print('\nPure BA optimization (no factors from the recovery used)')
print(row_format.format(*datasets_short))

print(row_format.format(*pure_ba['ate']))
#print(row_format.format(*pure_ba['time']))
print(row_format.format(*pure_ba['num_frames']))


