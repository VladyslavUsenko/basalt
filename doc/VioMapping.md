## EuRoC dataset

We demonstrate the usage of the system with the `MH_05_difficult` sequence of the [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) as an example.

**Note:** The path to calibration and configuration files used here works for the APT installation. If you compile from source specify the appropriate path to the files in [data folder](/data/).

Download the sequence from the dataset and extract it. 
```
mkdir euroc_data
cd euroc_data
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip
mkdir MH_05_difficult
cd MH_05_difficult
unzip ../MH_05_difficult.zip
cd ../
```

### Visual-inertial odometry
To run the visual-inertial odometry execute the following command in `euroc_data` folder where you downloaded the dataset.
```
basalt_vio --dataset-path MH_05_difficult/ --cam-calib /usr/etc/basalt/euroc_ds_calib.json --dataset-type euroc --config-path /usr/etc/basalt/euroc_config.json --marg-data euroc_marg_data --show-gui 1 
```
The command line options have the following meaning:
* `--dataset-path` path to the dataset.
* `--dataset-type` type of the datset. Currently only `bag` and `euroc` formats of the datasets are supported.
* `--cam-calib` path to camera calibration file. Check [calibration instructions](doc/Calibration.md) to see how the calibration was generated.
* `--config-path` path to the configuration file.
* `--marg-data` folder where the data from keyframe marginalization will be stored. This data can be later used for visual-inertial mapping.
* `--show-gui` enables or disables GUI.

This opens the GUI and runs the sequence. The processing happens in the background as fast as possible, and the visualization results are saved in the GUI and can be analysed offline.
![MH_05_VIO](/doc/img/MH_05_VIO.png)

The buttons in the GUI have the following meaning:
* `show_obs` toggles the visibility of the tracked landmarks in the image view.
* `show_ids` toggles the IDs of the points.
* `show_est_pos` shows the plot of the estimated position.
* `show_est_vel` shows the plot of the estimated velocity.
* `show_est_bg` shows the plot of the estimated gyro bias.
* `show_est_ba` shows the plot of the estimated accel bias.
* `show_gt` shows ground-truth trajectory in the 3D view.

By default the system starts with `continue_fast` enabled. This option visualizes the latest processed frame until the end of the sequence. Alternatively, the `continue` visualizes every frame without skipping. If both options are disabled the system shows the frame that is selected with the `show_frame` slider and the user can move forward and backward with `next_step` and `prev_step` buttons. The `follow` button changes between the static camera and the camera attached to the current frame.

For evaluation the button `align_se3` is used. It aligns the GT trajectory with the current estimate using an SE(3) transformation and prints the transformation and the root-mean-squared absolute trajectory error (RMS ATE).

The button `save_traj` saves the trajectory in one of two formats (`euroc_fmt` or `tum_rgbd_fmt`). In EuRoC format each pose is a line in the file and has the following format `timestamp[ns],tx,ty,tz,qw,qx,qy,qz`. TUM RBG-D can be used with [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools) or [UZH](https://github.com/uzh-rpg/rpg_trajectory_evaluation) trajectory evaluation tools and has the following format `timestamp[s] tx ty tz qx qy qz qw`. 


### Visual-inertial mapping
To run the mapping tool execute the following command:
```
basalt_mapper --cam-calib /usr/etc/basalt/euroc_ds_calib.json --marg-data euroc_marg_data
```
Here `--marg-data` is the folder with the results from VIO.

This opens the GUI and extracts non-linear factors from the marginalization data.
![MH_05_MAPPING](/doc/img/MH_05_MAPPING.png)

The buttons in the GUI have the following meaning:
* `show_frame1`, `show_cam1`, `show_frame2`, `show_cam2` allows you to assign images to image view 1 and 2 from different timestamps and cameras.
* `show_detected` shows the detected keypoints in the image view window.
* `show_matches` shows feature matching results.
* `show_inliers` shows inlier matches after geometric verification.
* `show_ids` prints point IDs. Can be used to find the same point in two views to check matches and inliers.
* `show_gt` shows the ground-truth trajectory.
* `show_edges` shows the edges from the factors. Relative-pose factors in red, roll-pitch factors in magenta and bundle adjustment co-visibility edges in green.
* `show_points` shows 3D landmarks.

The workflow for the mapping is the following:
* `detect` detect the keypoints in the keyframe images.
* `match` run the geometric 2D to 2D matching between image frames.
* `tracks` build tracks from 2D matches and triangulate the points.
* `optimize` run the optimization.
* `align_se3` align ground-truth trajectory in SE(3) and print the transformation and the error.

The `num_opt_iter` slider controls the maximum number of iterations executed when pressing `optimize`.

The button `save_traj` works similar to the VIO, but saves the keyframe trajectory (subset of frames).

For more systematic evaluation see the evaluation scripts in the [scripts/eval_full](/scripts/eval_full) folder.

**NOTE: It appears that only the datasets in ASL Dataset Format (`euroc` dataset type in our notation) contain ground truth that is time-aligned to the IMU and camera images. It is located in the `state_groundtruth_estimate0` folder. Bag files have raw Mocap measurements that are not time aligned and should not be used for evaluations.**



### Optical Flow
The visual-inertial odometry relies on the optical flow results. To enable a better analysis of the system we also provide a separate optical flow executable
```
basalt_opt_flow --dataset-path MH_05_difficult/ --cam-calib /usr/etc/basalt/euroc_ds_calib.json --dataset-type euroc --config-path /usr/etc/basalt/euroc_config.json --show-gui 1
```

This will run the GUI and print an average track length after the dataset is processed.
![MH_05_OPT_FLOW](/doc/img/MH_05_OPT_FLOW.png)


## TUM-VI dataset

We demonstrate the usage of the system with the `magistrale1` sequence of the [TUM-VI dataset](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) as an example.

Download the sequence from the dataset and extract it. 
```
mkdir tumvi_data
cd tumvi_data
wget http://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-magistrale1_512_16.tar
tar -xvf dataset-magistrale1_512_16.tar
```

### Visual-inertial odometry
To run the visual-inertial odometry execute the following command in `tumvi_data` folder where you downloaded the dataset.
```
basalt_vio --dataset-path dataset-magistrale1_512_16/ --cam-calib /usr/etc/basalt/tumvi_512_ds_calib.json --dataset-type euroc --config-path /usr/etc/basalt/tumvi_512_config.json --marg-data tumvi_marg_data --show-gui 1 
```
![magistrale1_vio](/doc/img/magistrale1_vio.png)

### Visual-inertial mapping
To run the mapping tool execute the following command:
```
basalt_mapper --cam-calib /usr/etc/basalt/tumvi_512_ds_calib.json --marg-data tumvi_marg_data
```
![magistrale1_mapping](/doc/img/magistrale1_mapping.png)
