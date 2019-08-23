#!/bin/bash

set -e
set -x

DATASET_PATH=/data/kitti_odom_grey/sequences

DATASETS=(00 02 03 04 05 06 07 08 09 10)


folder_name=eval_results_kitti
mkdir $folder_name

for d in ${DATASETS[$CI_NODE_INDEX-1]}; do
echo $d
   basalt_vio --dataset-path  $DATASET_PATH/$d --cam-calib $DATASET_PATH/$d/basalt_calib.json \
        --dataset-type kitti --show-gui 0 --config-path /usr/etc/basalt/kitti_config.json --result-path $folder_name/vo_$d --save-trajectory kitti --use-imu 0

   mv trajectory_kitti.txt $folder_name/kitti_$d.txt

   basalt_kitti_eval --traj-path $folder_name/kitti_$d.txt --gt-path $DATASET_PATH/$d/poses.txt --result-path $folder_name/rpe_$d.txt

done

