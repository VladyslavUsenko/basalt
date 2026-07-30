# Context Directory

This directory contains LLM-generated reference documents for the Basalt SLAM project.
These serve as long-term memory to avoid re-investigating the same questions.

## Documents

| File | Topic | Date |
|------|-------|------|
| `gt_slam_alignment.md` | GT vs SLAM alignment analysis, gravity-alignment explanation, frame geometry, two-part fix — EuRoC + TUM-VI | 2026-07-09/11 |
| `euroc_coordinate_frames.md` | EuRoC dataset coordinate frame reference, GT column decoding | 2026-07-09 |
| `tumvi_coordinate_frames.md` | TUM-VI dataset coordinate frame reference, GT column decoding | 2026-07-11 |
| `vio_localmapper_correction_loop.md` | VIO drift vs local BA correction loop; why live trajectory is never corrected; gauge freedom in NfrMapper BA; path to fix | 2026-07-11 |
| `airsim_camera_extrinsics.md` | AirSim → Basalt T_imu_cam derivation; NED↔CV axis permutation; SE3 composition and inversion; front_center numerical result | 2026-07-13 |
| `imu_noise_parameters.md` | CosysAirSim IMU noise params → Basalt conversion; ARW/VRW/FOGM bias model; phone-class SITL values; Allan deviation reference | 2026-07-13 |
| `imu_static_calibration.md` | calib_accel_bias (9-param) and calib_gyro_bias (12-param) model; code usage; EuRoC vs SITL values; how to calibrate a real IMU | 2026-07-13 |

## Quick Reference

### GT-SLAM Mismatch Fix (applies to both EuRoC and TUM-VI)
- **Root cause**: World frame origin/orientation mismatch (NOT camera-IMU extrinsics)
- **Fix**: SE(3) first-pose alignment applied in `dataset_io_euroc.h` at IO read time
- **Formula**: `T_gt_aligned[i] = T_gt[0].inverse() * T_gt[i]`

### EuRoC GT Frame
- `state_groundtruth_estimate0` → GT is `T_w_i` (body=IMU, T_BS=Identity confirmed)
- No camera-IMU extrinsics needed for this GT source
- `mocap0` (raw MoCap) → GT is in marker frame; needs `T_imu_marker` (but `T_imu_marker=I` in ds_calib)

### TUM-VI GT Frame
- GT file: `mocap0/data.csv` — **no `state_groundtruth_estimate0`** directory
- GT is `T_w_i` (already converted to IMU frame in EuRoC export)
- **Proof**: `dso/gt_imu.csv` has identical values and explicitly labels the IMU frame
- No sensor.yaml files; calibration via `tumvi_512_ds_calib.json`
- Only room sequences have full-trajectory GT
- 7-column format (no velocity/bias), ~120 Hz Vicon rate

### Downloaded Dataset
- EuRoC: `data/machine_hall/MH_01_easy/`, `data/machine_hall/MH_05_difficult/`
- TUM-VI: `data/TUM/dataset-room1_512_16/` (room1, 512×16 EuRoC export, 1.78 GB)

### AirSim SITL Calibration (`data/sitl_calib.json`)
- **T_imu_cam**: analytically derived — see `airsim_camera_extrinsics.md`
  - front_center (X=0.5, Y=0, Z=0.1, Pitch=−45°): `qx=0.2706, qy=0.2706, qz=0.6533, qw=0.6533`
- **IMU noise**: phone-class MEMS, 200 Hz — see `imu_noise_parameters.md`
  - `gyro_noise_std=5.82e-4`, `accel_noise_std=0.020`, `gyro_bias_std=7.92e-6`, `accel_bias_std=2.83e-4`
- **calib_accel_bias / calib_gyro_bias**: ALL ZEROS for simulation — see `imu_static_calibration.md`
  - Previously had EuRoC ADIS16448 values incorrectly copied — fixed to zeros

### IMU Static Calibration Key Facts
- `calib_accel_bias`: 9 params `[b_x, b_y, b_z, s1–s6 (lower-triangular scale/misalignment)]`
- `calib_gyro_bias`: 12 params `[b_x, b_y, b_z, s1–s9 (full 3×3 scale/misalignment)]`
- Model: `x_calibrated = (I + S) · x_raw − b`  (applied per sample, `sqrt_keypoint_vio.cpp:171-173`)
- Bias vector (first 3) seeds VIO initial dynamic bias: `basalt_slam.cpp:213-214`
- For SITL: all zero. For real hardware: run `basalt_calibrate_imu`
