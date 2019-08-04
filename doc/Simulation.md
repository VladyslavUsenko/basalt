## Simulator

For better evaluation of the system we use the simulated environment where the optical flow and IMU data is generated from the ground truth by adding noise.

**Note:** The path to calibration and configuration files used here works for the APT installation. If you compile from source specify the appropriate path to the files in [data folder](/data/).


### Visual-inertial odometry simulator
```
basalt_vio_sim --cam-calib /usr/etc/basalt/euroc_ds_calib.json --marg-data sim_marg_data --show-gui 1 
```

The command line options have the following meaning:
* `--cam-calib` path to camera calibration file. Check [calibration instructions](doc/Calibration.md) to see how the calibration was generated.
* `--marg-data` folder where the data from keyframe marginalization will be stored. This data can be later used for visual-inertial mapping simulator.
* `--show-gui` enables or disables GUI.

This opens the GUI and runs the sequence.
![SIM_VIO](/doc/img/SIM_VIO.png)

The buttons in the GUI have the following meaning:
* `show_obs` toggles the visibility of the ground-truth landmarks in the image view.
* `show_obs_noisy` toggles the visibility of the noisy landmarks in the image view.
* `show_obs_vio` toggles the visibility of the landmarks estimated by VIO in the image view.
* `show_ids` toggles the IDs of the landmarks.
* `show_accel` shows noisy accelerometer measurements generated from the ground-truth spline.
* `show_gyro` shows noisy gyroscope measurements generated from the ground-truth spline.
* `show_gt_...` shows ground truth position, velocity and biases.
* `show_est_...` shows VIO estimates of the position, velocity and biases.
* `next_step` proceeds to next frame.
* `continue` plays the sequence.
* `align_se3` performs SE(3) alignment with ground-truth trajectory and prints the RMS ATE to the console.


### Visual-inertial mapping simulator
```
basalt_mapper_sim --cam-calib /usr/etc/basalt/euroc_ds_calib.json --marg-data sim_marg_data --show-gui 1
```
The command line arguments are the same as above.

This opens the GUI where the map can be processed.
![SIM_MAPPER](/doc/img/SIM_MAPPER.png)

The system processes the marginalization data and extracts the non-linear factors from them. Roll-pitch and relative-pose factors are initially added to the system. One way to verify that they result in gravity-aligned map is the following
* `optimize` runs the optimization
* `rand_inc` applies a random increment to all frames of the system. If you run the `optimize` until convergence afterwards, and press `align_se3` the alignment transformation should only contain the rotation around Z axis.
* `rand_yaw` applies an increment in yaw to all poses. This should not change the error of the optimization once is have converged.
* `setup_points` triangulates the points and adds them to optimization. You should optimize the system again after adding the points.
* `align_se3` performs SE(3) alignment with ground-truth trajectory and prints the RMS ATE to the console.

For comparison we also provide the `basalt_mapper_sim_naive` executable that has the same parameters. It runs a global bundle-adjustment on keyframe data and inserts pre-integrated IMU measurements between keyframes. This executable is included for comparison only.