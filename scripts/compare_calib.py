#!/usr/bin/env python3


import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation


def print_abs_rel(info, v_0, v_1):
    diff = np.abs(np.linalg.norm(v_0 - v_1))
    out = f'{info}:\t{diff:.5f}'

    if diff < 10e-7:
        out += ' (0.0%)'
    else:
        out += f' ({diff / (np.abs(np.linalg.norm(v_0)) * 100.0):.7f}%)'

    print(out)


def main(calib_path_1, calib_path_2):
    with open(calib_path_1, 'r') as c_1, open(calib_path_2, 'r') as c_2:
        calib0 = json.load(c_1)
        calib1 = json.load(c_2)

    for i, (t_imu_cam_0, t_imu_cam_1) in enumerate(
            zip(calib0['value0']['T_imu_cam'], calib1['value0']['T_imu_cam'])):
        print(f'\nCamera {i} transformation differences')
        t_0 = np.array(list(t_imu_cam_0.values())[0:2])
        t_1 = np.array(list(t_imu_cam_1.values())[0:2])
        r_0 = Rotation(list(t_imu_cam_0.values())[3:7])
        r_1 = Rotation(list(t_imu_cam_1.values())[3:7])

        print_abs_rel(f'Transformation', t_0, t_1)
        print_abs_rel(f'Rotation', r_0.as_rotvec(), r_1.as_rotvec())

    for i, (intrinsics0, intrinsics1) in enumerate(
            zip(calib0['value0']['intrinsics'], calib1['value0']['intrinsics'])):
        print(f'\nCamera {i} intrinsics differences')

        for (
                k_0, v_0), (_, v_1) in zip(
                intrinsics0['intrinsics'].items(), intrinsics1['intrinsics'].items()):
            print_abs_rel(f'Difference for {k_0}', v_0, v_1)

    print_abs_rel('\nAccel Bias Difference',
                  np.array(calib0['value0']['calib_accel_bias'][0:2]),
                  np.array(calib1['value0']['calib_accel_bias'][0:2]))

    print_abs_rel('Accel Scale Difference',
                  np.array(calib0['value0']['calib_accel_bias'][3:9]),
                  np.array(calib1['value0']['calib_accel_bias'][3:9]))

    print_abs_rel('Gyro Bias Difference',
                  np.array(calib0['value0']['calib_gyro_bias'][0:2]),
                  np.array(calib1['value0']['calib_gyro_bias'][0:2]))

    print_abs_rel('Gyro Scale Difference',
                  np.array(calib0['value0']['calib_gyro_bias'][3:12]),
                  np.array(calib1['value0']['calib_gyro_bias'][3:12]))

    print_abs_rel(
        '\nAccel Noise Std Difference',
        calib0['value0']['accel_noise_std'],
        calib1['value0']['accel_noise_std'])
    print_abs_rel(
        'Gyro Noise Std Difference',
        calib0['value0']['gyro_noise_std'],
        calib1['value0']['gyro_noise_std'])
    print_abs_rel(
        'Accel Bias Std Difference',
        calib0['value0']['accel_bias_std'],
        calib1['value0']['accel_bias_std'])
    print_abs_rel(
        'Gyro Bias Std Difference',
        calib0['value0']['gyro_bias_std'],
        calib1['value0']['gyro_bias_std'])

    print_abs_rel(
        '\nCam Time Offset Difference',
        calib0['value0']['cam_time_offset_ns'],
        calib0['value0']['cam_time_offset_ns'])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib_path_1')
    parser.add_argument('calib_path_2')

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()

    main(args.calib_path_1, args.calib_path_2)
