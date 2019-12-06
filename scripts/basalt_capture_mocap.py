#!/usr/bin/env python

import sys
import os
import rospy
import argparse
from geometry_msgs.msg import TransformStamped


def callback(data):
    global out_file, time_offset
    if not out_file:
        return

    if not time_offset:
        time_offset = rospy.Time().now() - data.header.stamp

    out_file.write('{},{},{},{},{},{},{},{}\n'.format(
        data.header.stamp + time_offset,
        data.transform.translation.x,
        data.transform.translation.y,
        data.transform.translation.z,
        data.transform.rotation.w,
        data.transform.rotation.x,
        data.transform.rotation.y,
        data.transform.rotation.z
    ))


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/vrpn_client/raw_transform', TransformStamped, callback)

    rospy.spin()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Record Motion Capture messages from ROS (/vrpn_client/raw_transform).')
    parser.add_argument('-d', '--dataset-path', required=True, help="Path to store the result")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    out_file = None
    time_offset = None

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    out_file = open(dataset_path + '/data.csv', 'w')
    out_file.write('#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n')
    listener()
    out_file.close()

