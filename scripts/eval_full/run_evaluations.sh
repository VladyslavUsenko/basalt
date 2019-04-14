#!/bin/bash

set -e
set -x

DATASET_PATH=/data/euroc

DATASETS=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult)


folder_name=eval_$(date +%Y%m%d_%H%M%S)
mkdir $folder_name

cp ../../data/euroc_config.json $folder_name/config.json


for d in ${DATASETS[@]}; do
    ../../build/basalt_vio --dataset-path  $DATASET_PATH/$d --cam-calib ../../data/euroc_ds_calib.json \
        --dataset-type euroc --show-gui 0 --config-path $folder_name/config.json \
        --result-path $folder_name/vio_$d --marg-data $folder_name/marg_$d/

    ../../build/basalt_mapper --show-gui 0 --cam-calib ../../data/euroc_ds_calib.json --marg-data $folder_name/marg_$d/ \
        --vocabulary ../../thirdparty/DBoW3/vocab/orbvoc.dbow3 --result-path $folder_name/mapper_$d
done

./gen_results.py $folder_name
