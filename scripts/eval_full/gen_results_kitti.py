#!/usr/bin/env python3

import os
import sys
import json


datasets = ['Seq.', '00', '02', '03','04', '05', '06','07', '08', '09', '10']
lengths = [100, 200, 300, 400, 500, 600, 700, 800]

# Other results.


vo = {
'trans_error': {},
'rot_error': {}
}

for l in lengths:
        vo['trans_error'][l] = ['Trans. error [%] ' + str(l) + 'm.']
        vo['rot_error'][l] = ['Rot. error [deg/m] ' + str(l) + 'm.']


out_dir = sys.argv[1]

mean_values = {
'mean_trans_error' : 0.0,
'mean_rot_error' : 0.0,
'total_num_meas' : 0.0
}

def load_data(x, prefix, key, mean_values):
    fname = out_dir + '/' + prefix + '_' + key + '.txt'
    if os.path.isfile(fname): 
        with open(fname, 'r') as f:
            j = json.load(f)
            res = j['results']
            for v in lengths:
                num_meas = res[str(v)]['num_meas']
                trans_error = res[str(v)]['trans_error']
                rot_error = res[str(v)]['rot_error']
                x['trans_error'][int(v)].append(round(trans_error, 5))
                x['rot_error'][int(v)].append(round(rot_error, 5))
                if num_meas > 0:
                        mean_values['mean_trans_error'] += trans_error*num_meas
                        mean_values['mean_rot_error'] += rot_error*num_meas
                        mean_values['total_num_meas'] += num_meas
    else:
        for v in lengths:
                x['trans_error'][int(v)].append(float('inf'))
                x['rot_error'][int(v)].append(float('inf'))

for key in datasets[1:]:
    load_data(vo, 'rpe', key, mean_values)


row_format ="{:>24}" + "{:>10}" * (len(datasets)-1)

datasets_short = [x[:5] for x in datasets]

print('\nVisual Odometry (Stereo)')
print(row_format.format(*datasets_short))

for l in lengths:
        print(row_format.format(*(vo['trans_error'][l])))

print()

for l in lengths:
        print(row_format.format(*(vo['rot_error'][l])))


print('Mean translation error [%] ',  mean_values['mean_trans_error']/mean_values['total_num_meas'])
print('Mean rotation error [deg/m] ', mean_values['mean_rot_error']/mean_values['total_num_meas'])