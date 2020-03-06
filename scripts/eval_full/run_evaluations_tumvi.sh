#!/bin/bash

set -e
set -x

DATASET_PATH=/data/tumvi/512_16/

DATASETS=(
dataset-corridor1_512_16
dataset-magistrale1_512_16
dataset-room1_512_16
dataset-slides1_512_16
)


folder_name=eval_results_tumvi
mkdir $folder_name



for d in ${DATASETS[$CI_NODE_INDEX-1]}; do
   basalt_vio --dataset-path  $DATASET_PATH/$d --cam-calib /usr/etc/basalt/tumvi_512_eucm_calib.json \
        --dataset-type euroc --show-gui 0 --config-path /usr/etc/basalt/tumvi_512_config.json \
        --result-path $folder_name/vio_$d --save-trajectory tum

   mv trajectory.txt $folder_name/${d}_basalt_poses.txt

done

#./gen_results_tumvi.py $folder_name > euroc_tumvi.txt