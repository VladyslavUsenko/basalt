#!/usr/bin/env python

import os
import sys
import json


datasets = ['Sequence', 'MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult',
'MH_05_difficult', 'V1_01_easy', 'V1_02_medium',
'V1_03_difficult', 'V2_01_easy', 'V2_02_medium']


# Other results.
results_vio = ['VIO RMS ATE [m]']
time_vio = ['VIO Time [s]']
num_frames_vio = ['VIO Num. Frames'] 

results_mapping = ['MAP RMS ATE [m]']
time_mapping = ['MAP Time [s]']
num_frames_mapping = ['MAP Num. KFs'] 

out_dir = sys.argv[1]


for key in datasets[1:]:
    fname = out_dir + '/vio_' + key
    if os.path.isfile(fname): 
        with open(fname, 'r') as f:
            j = json.load(f)
            res = round(j['rms_ate'], 3)
            results_vio.append(float(res))
            time_vio.append(round(j['exec_time_ns']*1e-9, 3))
            num_frames_vio.append(j['num_frames'])
    else:
        results_vio.append(float('Inf'))
        time_vio.append(float('Inf'))
        num_frames_vio.append(float('Inf'))
    
    fname = out_dir + '/mapper_' + key
    if os.path.isfile(fname): 
        with open(fname, 'r') as f:
            j = json.load(f)
            res = round(j['rms_ate'], 3)
            results_mapping.append(float(res))
            time_mapping.append(round(j['exec_time_ns']*1e-9, 3))
            num_frames_mapping.append(j['num_frames'])
    else:
        results_mapping.append(float('Inf'))
        time_mapping.append(float('Inf'))
        num_frames_mapping.append(float('Inf'))

row_format ="{:>17}" * (len(datasets))
print row_format.format(*datasets)

print row_format.format(*results_vio)
print row_format.format(*time_vio)
print row_format.format(*num_frames_vio)

print '\n'

print row_format.format(*datasets)

print row_format.format(*results_mapping)
print row_format.format(*time_mapping)
print row_format.format(*num_frames_mapping)

