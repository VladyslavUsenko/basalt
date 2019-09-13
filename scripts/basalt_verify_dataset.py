#!/usr/bin/env python3

import sys
import math
import os
import argparse

import numpy as np


parser = argparse.ArgumentParser(description='Check the dataset. Report if any images are missing.')
parser.add_argument('-d', '--dataset-path', required=True, help="Path to the dataset in Euroc format")
args = parser.parse_args()


dataset_path = args.dataset_path

print(dataset_path)

timestamps = {}
exposures = {}

for sensor in ['cam0', 'cam1', 'imu0']:
    data = np.loadtxt(dataset_path + '/mav0/' + sensor + '/data.csv', usecols=[0], delimiter=',', dtype=np.int64)
    timestamps[sensor] = data

# check if dataset is OK...
for key, value in timestamps.items():
    times = value * 1e-9
    min_t = times.min()
    max_t = times.max()
    interval = max_t - min_t
    diff = times[1:] - times[:-1]
    print('==========================================')
    print('sensor', key)
    print('min timestamp', min_t)
    print('max timestamp', max_t)
    print('interval', interval)
    print('hz', times.shape[0] / interval)
    print('min time between consecutive msgs', diff.min())
    print('max time between consecutive msgs', diff.max())
    for i, d in enumerate(diff):
        # Note: 0.001 is just a hacky heuristic, since we have nothing faster than 1000Hz. Should maybe be topic-specific.
        if d < 0.001:
            print("ERROR: Difference on consecutive measurements too small: {} - {} = {}".format(times[i + 1], times[i],
                                                                                                 d) + ' in sensor ' + key)

# check if we have all images for timestamps
timestamp_to_topic = {}

for key, value in timestamps.items():
    if not key.startswith('cam'):
        continue
    for v in value:
        if v not in timestamp_to_topic:
            timestamp_to_topic[v] = list()
        timestamp_to_topic[v].append(key)

for key in timestamp_to_topic.keys():
    if len(timestamp_to_topic[key]) != 2:
        print('timestamp', key, 'has topics', timestamp_to_topic[key])

# check image data.
img_extensions = ['.png', '.jpg', '.webp']
for key, value in timestamps.items():
    if not key.startswith('cam'):
        continue
    for v in value:
        path = dataset_path + '/mav0/' + key + '/data/' + str(v)
        img_exists = False
        for e in img_extensions:
            if os.path.exists(dataset_path + '/mav0/' + key + '/data/' + str(v) + e):
                img_exists = True

        if not img_exists:   
            print('No image data for ' + key + ' at timestamp ' + str(v))
    
    exposure_file = dataset_path + '/mav0/' + key + '/exposure.csv'
    if not os.path.exists(exposure_file):
        print('No exposure data for ' + key)
        continue
    
    exposure_data = np.loadtxt(exposure_file, delimiter=',', dtype=np.int64)
    for v in value:
        idx = np.searchsorted(exposure_data[:, 0], v)
        if exposure_data[idx, 0] != v:
            print('No exposure data for ' + key + ' at timestamp ' + str(v))
