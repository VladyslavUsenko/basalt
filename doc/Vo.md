## KITTI dataset

[![teaser](/doc/img/kitti_video.png)](https://www.youtube.com/watch?v=M_ZcNgExUNc)

We demonstrate the usage of the system with the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) as an example.

**Note:** The path to calibration and configuration files used here works for the APT installation. If you compile from source specify the appropriate path to the files in [data folder](/data/).

Download the sequences (`data_odometry_gray.zip`) from the dataset and extract it. 
```
# We assume you have extracted the sequences in ~/dataset_gray/sequences/
# Convert calibration to the basalt format
basalt_convert_kitti_calib.py ~/dataset_gray/sequences/00/

# If you want to convert calibrations for all sequences use the following command
for i in {00..21}; do basalt_convert_kitti_calib.py ~/dataset_gray/sequences/$i/; done
```
Optionally you can also copy the provided ground-truth poses to `poses.txt` in the corresponding sequence.

### Visual odometry
To run the visual odometry execute the following command.
```
basalt_vio --dataset-path ~/dataset_gray/sequences/00/ --cam-calib /work/kitti/dataset_gray/sequences/00/basalt_calib.json --dataset-type kitti --config-path /usr/etc/basalt/kitti_config.json --show-gui 1 --use-imu 0
```
![magistrale1_vio](/doc/img/kitti.png)
