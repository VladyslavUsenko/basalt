#!/bin/bash

set -e
set -x

DATASET_PATH=/data/euroc

DATASETS=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium)


folder_name=eval_results
mkdir $folder_name



for d in ${DATASETS[$CI_NODE_INDEX-1]}; do
   basalt_vio --dataset-path  $DATASET_PATH/$d --cam-calib /usr/etc/basalt/euroc_eucm_calib.json \
        --dataset-type euroc --show-gui 0 --config-path /usr/etc/basalt/euroc_config.json \
        --result-path $folder_name/vio_$d --marg-data eval_tmp_marg_data --save-trajectory tum

   mv trajectory.txt $folder_name/traj_vio_$d.txt

    basalt_mapper --show-gui 0 --cam-calib /usr/etc/basalt/euroc_eucm_calib.json --config-path /usr/etc/basalt/euroc_config.json --marg-data eval_tmp_marg_data \
        --result-path $folder_name/mapper_$d

    basalt_mapper --show-gui 0 --cam-calib /usr/etc/basalt/euroc_eucm_calib.json --config-path /usr/etc/basalt/euroc_config_no_weights.json --marg-data eval_tmp_marg_data \
        --result-path $folder_name/mapper_no_weights_$d
    
        basalt_mapper --show-gui 0 --cam-calib /usr/etc/basalt/euroc_eucm_calib.json --config-path /usr/etc/basalt/euroc_config_no_factors.json --marg-data eval_tmp_marg_data \
        --result-path $folder_name/mapper_no_factors_$d
    
    rm -rf eval_tmp_marg_data
done

#./gen_results.py $folder_name > euroc_results.txt
