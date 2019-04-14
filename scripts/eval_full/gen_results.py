#!/usr/bin/env python

import os
import sys
import numpy as np


datasets = ['MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult',
'MH_05_difficult', 'V1_01_easy', 'V1_02_medium',
'V1_03_difficult', 'V2_01_easy', 'V2_02_medium']


# Other results.
results_vio = []
results_mapping = []

out_dir = sys.argv[1]


for key in datasets:
    fname = out_dir + '/vio_' + key
    if os.path.isfile(fname): 
        res = round(float(np.loadtxt(fname)), 3)
        results_vio.append(float(res))
    else:
        results_vio.append(float('Inf'))
    
    fname = out_dir + '/mapper_' + key
    if os.path.isfile(fname): 
        res = round(float(np.loadtxt(fname)), 3)
        results_mapping.append(float(res))
    else:
        results_mapping.append(float('Inf'))

row_format ="{:>17}" * (len(datasets))
print row_format.format(*datasets)
print row_format.format(*results_vio)
print row_format.format(*results_mapping)

