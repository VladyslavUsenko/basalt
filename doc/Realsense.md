# Tutorial on Camera-IMU and Motion Capture Calibration with Realsense T265

![Realsense](/doc/img/realsense_setup.jpg)

In this tutorial we explain how to perform photometric and geometric calibration of the multi-camera setup and then calibrate the transformations between cameras, IMU and the motion capture marker setup. To make sure the calibration is successful we recommend to rigidly attach markers to the camera as shown on the figure above.

## Dataset
We provide a set of example datasets for the calibration. Even if you plan to record your own datasets (covered in the next section), we recommend to download and try the provided examples:
```
mkdir ~/t265_calib_data
cd ~/t265_calib_data
wget http://vision.in.tum.de/tumvi/t265/response_calib.zip
wget http://vision.in.tum.de/tumvi/t265/cam_calib.zip
wget http://vision.in.tum.de/tumvi/t265/imu_calib.zip
wget http://vision.in.tum.de/tumvi/t265/sequence0.zip
unzip response_calib
unzip cam_calib
unzip imu_calib
unzip sequence0
```

## Recording Own Dataset
You can record your own sequences using the `basalt_rs_t265_record` executable:
```
basalt_rs_t265_record --dataset-path ~/t265_calib_data/ --manual-exposure 
```
* `--dataset-path` specifies the location where the recorded dataset will be stored. In this case it will be stored in `~/t265_calib_data/<current_timestamp>/`.
* `--manual-exposure` disables the autoexposure. In this tutorial the autoexposure is disabled for all calibration sequences, but for the VIO sequence (sequence0) we enable it.

![t265_record](/doc/img/t265_record.png)

The GUI elements have the following meaning:
* `webp_quality` compression quality. The highest value (101) means lossless compression. For photometric calibration it is important not to have any compression artifacts, so we record these calibration sequences with lossless compression.
* `skip_frames` reduces the framerate of the recorded dataset by skipping frames.
* `exposure` controls the exposure time of the cameras.
* `record` starts/stops the recoding of the dataset. If you run the system on the slow PC pay attention to the number of messages in the queue. If it goes beyond the limit the recorder will start dropping frames.
* `export_calibration` exports factory calibration in the basalt format.

After recoding the dataset it is a good practice to verify the quality of the dataset. You can do this by running:
```
basalt_verify_dataset.py -d ~/t265_calib_data/<dataset_path>/
```
It will report the actual frequencies of the recorded sensor messages and warn you if any files with image data are missing.

Every sequence required for the calibration should have certain properties to enable successful calibration. Pay attention to the **Important for recording the dataset** subsections and inspect the provided examples before recoding your own sequences.

## Response calibration
In this project we assume a camera has a linear response function (intensity of the pixel is linearly proportional to the amount of light captured by the pixel). In this section we will verify this for the Realsense T265 cameras. We will need to record a static scene with different exposure times.

**Important for recording the dataset:**
* Keep the camera static and make sure that nothing in the scene moves during the recording.
* Move `webp_quality` slider to the highest value to enable lossless compression.
* Optionally set the `skip_frames` slider to 3-5 to speed up the dataset recoding.
* Start recoding and slowly move the exposure slider up and down. Stop recoding.
* Rename the dataset to `response_calib`.

Run the response function calibration:
```
basalt_response_calib.py -d ~/t265_calib_data/response_calib
```
You should see the response function and the irradiance image similar to the one shown below. For the details of the algorithm see Section 2.3.1 of [[arXiv:1607.02555]](https://arxiv.org/abs/1607.02555). The results suggest that the response function used in the camera is linear. 
![t265_inv_resp_irradiance](/doc/img/t265_inv_resp_irradiance.png)

## Multi-Camera Geometric and Vignette Calibration
For the camera calibration we need to record a dataset with a static aprilgrid pattern.

**Important for recording the dataset:**
* Move `webp_quality` slider to the highest value to enable lossless compression (important for vignette calibration).
* Set the `skip_frames` slider to 5 to speed up the dataset recoding.
* Move the camera slowly to reduce the motion blur.
* Cover the entire field of view of the camera with the calibration pattern. Try to observe the pattern from different angles.
* Make sure you do not cast shadows at the pattern (important for vignette calibration).
* Rename the dataset to `cam_calib` 

Run the calibration executable:
```
basalt_calibrate --dataset-path ~/t265_calib_data/cam_calib --dataset-type euroc --result-path ~/t265_calib_results/ --aprilgrid /usr/etc/basalt/aprilgrid_6x6.json --cam-types kb4 kb4
```
To perform the calibration follow these steps:
* `load_dataset` load the dataset.
* `detect_corners` detect corners. If the corners were detected before the cached detections will be loaded at the previous step so there is no need to re-run the detection.
* `init_cam_intr` initialize camera intrinsics.
* `init_cam_poses` initialize camera poses using the current intrinsics estimate.
* `init_cam_extr` initialize transformations between multiple cameras.
* `init_opt` initialize optimization.
* `opt_until_converge` optimize until convergence.
* `init_cam_poses` some initial poses computed from the initialized intrinsics can be far from optimum and not converge to the right minimum. To improve the final result we can re-initialize poses with optimized intrinsics.
* `init_opt` initialize optimization with new initial poses.
* `opt_until_converge` optimize until convergence.
* `compute_vign` after optimizing geometric models compute the vignetting of the cameras.
* `save_calib` save calibration file to the `~/t265_calib_results/calibration.json`.

![t265_cam_calib](/doc/img/t265_cam_calib.png)


## IMU and Motion Capture Calibration
After calibrating cameras we can proceed to geometric and time calibration of the cameras, IMU and motion capture system. Setting up the motion capture system is specific for your setup. 

For the motion capture recording we use [ros_vrpn_client](https://github.com/ethz-asl/ros_vrpn_client) with [basalt_capture_mocap.py](/scripts/basalt_capture_mocap.py). We record the data to the `mocap0` folder and then move it to the `mav0` directory of the camera dataset. This script is provided as an example. Motion capture setup is different in every particular case.

**Important for recording the dataset:**
* Set the `skip_frames` slider to 1 to use the full framerate.
* Reduce the exposure time to reduce the motion blur.
* Move the setup such that all axes of accelerometer and gyro are excited. This means moving with acceleration along X, Y and Z axes and rotating around those axes.
* Do not forget to simultaneously record motion capture data.
* Rename the dataset to `imu_calib`.

Run the calibration executable:
```
basalt_calibrate_imu --dataset-path ~/t265_calib_data/imu_calib --dataset-type euroc --result-path ~/t265_calib_results/ --aprilgrid /usr/etc/basalt/aprilgrid_6x6.json --accel-noise-std 0.00818 --gyro-noise-std 0.00226 --accel-bias-std 0.01 --gyro-bias-std 0.0007
```

To perform the calibration follow these steps:
* `load_dataset` load the dataset.
* `detect_corners` detect corners. If the corners were detected before the cached detections will be loaded at the previous step so there is no need to re-run the detection.
* `init_cam_poses` initialize camera poses.
* `init_cam_imu` initialize the transformation between cameras and the IMU.
* `init_opt` initialize optimization.
* `opt_until_converge` optimize until convergence.
* Enable `opt_cam_time_offset` and `opt_imu_scale` to optimize the time offset between cameras and the IMU and the IMU scale.
* `opt_until_converge` optimize until convergence.
* `init_mocap` initialize transformation between the motion capture marker frame and the IMU and the transformation between the aprilgrid pattern and the motion capture system origin.
* Enable `opt_mocap`.
* `opt_until_converge` optimize until convergence.
* `save_calib` save calibration file to the `~/t265_calib_results/calibration.json`.
* `save_mocap_calib` save motion capture system calibration file to the `~/t265_calib_results/mocap_calibration.json`.

![t265_imu_calib](/doc/img/t265_imu_calib.png)


## Generating Time-Aligned Ground Truth
Since motion capture system and the PC where the dataset was recorded might not have the same clock we need to perform the time synchronization. Additionally we need to transform the coordinate frame of the GT data to the IMU frame (originally it is in the coordinate frame attached to the markers).

The raw motion capture data is stored in the `mav/mocap0/` folder. We can find the time offset by minimizing the error between gyro measurements and rotational velocities computed from the motion capture data. If you press the `save_aligned_dataset` button the resulting trajectory (time aligned and transformed to the IMU frame) will be written to `mav/gt/data.csv` and automatically loaded when available.
```
basalt_time_alignment --dataset-path ~/t265_calib_data/sequence0 --dataset-type euroc --calibration ~/t265_calib_results/calibration.json --mocap-calibration ~/t265_calib_results/mocap_calibration.json
```
You should be able to see that, despite some noise, rotational velocity computed from the motion capture data aligns well with gyro measurements.
![t265_time_align_gyro](/doc/img/t265_time_align_gyro.png)

You can also switch to the error function plot and see that there is a clear minimum corresponding to the computed time offset.
![t265_time_align_error](/doc/img/t265_time_align_error.png)

**Note:** If you want to run the time alignment again you should delete the `~/t265_calib_data/sequence0/mav/gt` folder first. If GT data already exist you will see the `save_aligned_dataset(disabled)` button which will **NOT** overwrite it.

## Running Visual-Inertial Odometry
Now we can run the visual-inertial odometry on the recorded dataset:
```
basalt_vio --dataset-path ~/t265_calib_data/sequence0 --cam-calib ~/t265_calib_results/calibration.json --dataset-type euroc --config-path /usr/etc/basalt/euroc_config.json --show-gui 1
```
After the system processes the whole sequence you can use `align_se3` button to align trajectory to the ground-truth data and compute RMS ATE.
![t265_vio](/doc/img/t265_vio.png)


## Running Visual-Inertial Odometry Live
It is also possible to run the odometry live with the camera. If no calibration files are provided the factory calibration will be used.
```
basalt_rs_t265_vio --cam-calib ~/t265_calib_results/calibration.json --config-path /usr/etc/basalt/euroc_config.json
```