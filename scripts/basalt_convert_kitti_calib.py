#!/usr/bin/env python3

import sys
import math
import numpy as np
import os
from string import Template
import cv2
import argparse

parser = argparse.ArgumentParser(description='Convert KITTI calibration to basalt and save it int the dataset folder as basalt_calib.json.')
parser.add_argument('-d', '--dataset-path', required=True, help="Path to the dataset in KITTI format")
args = parser.parse_args()

dataset_path = args.dataset_path

print(dataset_path)

kitti_calib_file = dataset_path + '/calib.txt'


calib_template = Template('''{
    "value0": {
        "T_imu_cam": [
            {
                "px": 0.0,
                "py": 0.0,
                "pz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0
            },
            {
                "px": $px,
                "py": 0.0,
                "pz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0
            }
        ],
        "intrinsics": [
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": $fx0,
                    "fy": $fy0,
                    "cx": $cx0,
                    "cy": $cy0
                }
            },
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": $fx1,
                    "fy": $fy1,
                    "cx": $cx1,
                    "cy": $cy1
                }
            }
        ],
        "resolution": [
            [
                $rx,
                $ry
            ],
            [
                $rx,
                $ry
            ]
        ],
        "vignette": [],
        "calib_accel_bias": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "calib_gyro_bias": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "imu_update_rate": 0.0,
        "accel_noise_std": [0.0, 0.0, 0.0],
        "gyro_noise_std": [0.0, 0.0, 0.0],
        "accel_bias_std": [0.0, 0.0, 0.0],
        "gyro_bias_std": [0.0, 0.0, 0.0],
        "cam_time_offset_ns": 0
    }
}
''')


with open(kitti_calib_file, 'r') as stream:
    lines = (' '.join([x.strip('\n ') for x in stream.readlines() if x.strip('\n ') ])).split(' ')

    if len(lines) != 52:
        print('Issues loading calibration')
        print(lines)
    
    P0 = np.array([float(x) for x in lines[1:13]]).reshape(3,4)
    P1 = np.array([float(x) for x in lines[14:26]]).reshape(3,4)
    print('P0\n', P0)
    print('P1\n', P1)

    tx = -P1[0,3]/P1[0,0]

    img = cv2.imread(dataset_path + '/image_0/000000.png')
    rx = img.shape[1]
    ry = img.shape[0]

    values = {'fx0': P0[0,0], 'fy0': P0[1,1], 'cx0': P0[0,2], 'cy0': P0[1,2], 'fx1': P1[0,0], 'fy1': P1[1,1], 'cx1': P1[0,2], 'cy1': P1[1,2], 'px': tx, 'rx': rx, 'ry': ry}

    calib = calib_template.substitute(values)
    print(calib)

    with open(dataset_path + '/basalt_calib.json', 'w') as stream2:
        stream2.write(calib)