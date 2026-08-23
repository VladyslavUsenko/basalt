# Context Directory

This directory contains LLM-generated reference documents for the Basalt SLAM project.
These serve as long-term memory to avoid re-investigating the same questions.

## How to update these files (MANDATORY)

These documents are incremental. Update them by correction and addition only, never by rewriting. Every fact already recorded must survive an edit, whether or not it bears on the task in hand, because the accumulated derivations, worked examples, code references and caveats are the whole value of the store and no single session would regenerate them.

When a recorded fact turns out to be wrong, rewrite it in place so that the document states the correct value as the only reading offered, then record the correction alongside it, naming what the file previously said, when it changed and why the old value was wrong. Never leave the wrong value standing as the primary text and never tag a block as superseded, because that leaves a reader to adjudicate between two competing claims, which is the largest source of confusion in a store of this kind. The correction note is what preserves the history and lets a reader identify an artefact built from the old value, so it must name that value explicitly, but it is a short note attached to a correct statement rather than a second version of the section. After editing, diff against the previous revision and confirm that every substantive line that vanished was deliberately corrected or removed rather than accidentally lost.

## Documents

| File | Topic | Date |
|------|-------|------|
| `gt_slam_alignment.md` | GT vs SLAM alignment analysis, gravity-alignment explanation, frame geometry, two-part fix — EuRoC + TUM-VI | 2026-07-09/11 |
| `euroc_coordinate_frames.md` | EuRoC dataset coordinate frame reference, GT column decoding | 2026-07-09 |
| `tumvi_coordinate_frames.md` | TUM-VI dataset coordinate frame reference, GT column decoding | 2026-07-11 |
| `vio_localmapper_correction_loop.md` | VIO drift vs local BA correction loop; why live trajectory is never corrected; gauge freedom in NfrMapper BA; path to fix; the 2026-08-23 conversion of the `out_state_queue` and `out_vis_queue` taps to `try_push` | 2026-07-11/2026-08-23 |
| `airsim_camera_extrinsics.md` | Project AirSim → Basalt T_imu_cam derivation; `origin`/`rpy-deg` z-y-x convention and pitch clamp; NED↔CV axis permutation; SE3 composition and inversion; front_center numerical result and quaternion check; proof the IMU sits at the parent-link origin; intrinsics; measured 7.42 Hz frame rate against a 19.6 Hz ceiling; the FLU extrinsic the bridge IMU topic would need | — |
| `imu_noise_parameters.md` | Project AirSim IMU noise params → Basalt, read directly in SI with no conversion; ARW/VRW theory and the external-source conversions; pure Wiener bias process and the σ_b/√τ mapping; current values, discrepancy table, and Part 7 on ArduPilot's added noise, 50 Hz filter and gyro drift | — |
| `ap_dds_imu_stream.md` | Live `/ap/imu/experimental/data` capture; NED frame confirmation; Basalt gravity auto-alignment path; the two ArduPilot stamp sources and why duplicate stamps are impossible at 167 Hz; the node's de-duplication filter; the inertial/visual epoch mismatch; outstanding measurements | — |
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

### Project AirSim SITL Calibration (`data/sitl_calib.json`)
- **T_imu_cam**: analytically derived and numerically verified — see `airsim_camera_extrinsics.md`
  - front_center (`"xyz": "0.5 0.0 0.1"`, `"rpy-deg": "0 -45 0"`): `qx=0.2706, qy=0.2706, qz=0.6533, qw=0.6533`
  - det=1, unit norm, optical axis 45° down from forward in NED
  - The IMU parses no `origin` and applies no lever-arm correction, so camera-to-body is camera-to-IMU exactly
- **Intrinsics**: `fx=fy=320, cx=320, cy=240` — correct for 90° HFOV at 640×480, pinhole, no distortion
- **Frame rate**: measured 7.42 Hz against a 19.6 Hz scene-tick ceiling, with intervals quantised to 51 ms and a worst gap of 1.64 s. `orb_slam3/config/Monocular/sitl.yaml:28` still declares `Camera.fps: 15` and overstates it twofold
- **IMU frame**: NED (`base_link_ned`), from ArduPilot AP_DDS, not the simulator's own IMU sensor — see `ap_dds_imu_stream.md`. No code change is needed to put the IMU in the frame Basalt expects
- **IMU noise**: three of four values are WRONG in the file — see `imu_noise_parameters.md`
  - `gyro_noise_std=5.818e-4` correct; `accel_noise_std` 1.70× too large; `gyro_bias_std` and `accel_bias_std` √2 too large
  - `imu_update_rate` is 167, correct, and equals the distinct-sample rate because no stamp ever repeats
- **calib_accel_bias / calib_gyro_bias**: ALL ZEROS for simulation — see `imu_static_calibration.md`
  - Correct because `turn-on-bias` is zero and no scale or misalignment error is modelled at all

### Project AirSim → Basalt IMU Conversions
Read from `core_sim/src/sensors/imu.cpp` and `core_sim/include/core_sim/sensors/imu.hpp`. Project AirSim reads SI directly and performs **no unit conversion whatever**, so the white-noise mappings are the identity.
- `gyro_noise_std  = gyroscope.angle-random-walk`  ← already rad/s/√Hz
- `accel_noise_std = accelerometer.velocity-random-walk`  ← already m/s²/√Hz, NOT mg and NOT m/s/√hr despite the name
- `gyro_bias_std   = gyroscope.bias-stability / √tau`  ← bias-stability already rad/s
- `accel_bias_std  = accelerometer.bias-stability / √tau`  ← bias-stability already m/s²
- The bias process is a plain Wiener random walk with coefficient σ_b/√τ. There is no mean-reversion term, so the FOGM factor √2 does not apply.
- No flag disables the noise; `ApplyNoiseModel` is called unconditionally at `imu.cpp:195`. Zero the parameters to get a clean IMU.
- Both an `accelerometer` and a `gyroscope` block must be present: `accelerometer_bias_stability_norm` is only assigned inside the loaders and is read uninitialised if the object is omitted.

### Timestamps on the ArduPilot IMU Topic
- Every `/ap/*` stamp comes from `AP_DDS_Client::update_topic(builtin_interfaces_msg_Time&)`, which uses the external `/clock` when `has_received_clock` is set and otherwise `AP::rtc().get_utc_usec()`.
- The 5 ms `AP_DDS_DELAY_IMU_TOPIC_MS` gate with a strict inequality gives a 6 ms period, exactly 166.667 Hz, measured with zero jitter across 69 consecutive stamps.
- Duplicate stamps are impossible on either branch, so the node's de-duplication filter at `node.cpp:190-211` drops nothing and `imu_update_rate` equals the message rate.
- **Open hazard**: a capture showed inertial stamps on a UTC epoch (~1.787e9 s) while camera stamps carry raw simulation time (~1.4e3 s), meaning `/clock` had not reached ArduPilot. Basalt cannot fuse streams with that offset. Verify with the epoch comparison in `ap_dds_imu_stream.md`.

### What Stands Between the Simulator and Basalt
Basalt subscribes to ArduPilot, not to the simulator. Three effects are added on the way, none configured by the simulator — see `imu_noise_parameters.md` Part 7.
- ArduPilot adds its own white noise at `AP_InertialSensor_SITL.cpp:114-120, 230-249`. Negligible, raising the densities by 1.0001 and 1.0005.
- `INS_ACCEL_FILTER` and `INS_GYRO_FILTER` are 50 Hz. Reduces visible scatter to 0.104 m/s² against the 0.152 the calibration implies, but leaves the preintegration covariance correct. Do **not** compensate.
- `gyro_drift()` at `:401-413` adds a deterministic 0.05 °/s per minute ramp on all three axes. This one matters: it exceeds the permitted bias excursion by 20× over a minute. Either set `SIM_DRIFT_SPEED` to 0 (preferred) or raise `gyro_bias_std` to 1.5586e-5.

### Basalt Gravity Convention
- World frame is Z-up, `g = (0,0,−9.81)` at `include/basalt/utils/imu_types.h:62`
- A NED (z-down) IMU is handled correctly, because `controller.cpp:163-167` routes an identity/zero init to the two-arg `initialize(bg, ba)`, leaving `initialized=false` so `sqrt_keypoint_vio.cpp:248` gravity-aligns via `FromTwoVectors(accel, UnitZ())`
- Side effect: the antiparallel case makes the initial heading arbitrary and non-reproducible between runs

### IMU Static Calibration Key Facts
- `calib_accel_bias`: 9 params `[b_x, b_y, b_z, s1–s6 (lower-triangular scale/misalignment)]`
- `calib_gyro_bias`: 12 params `[b_x, b_y, b_z, s1–s9 (full 3×3 scale/misalignment)]`
- Model: `x_calibrated = (I + S) · x_raw − b`  (applied per sample, `sqrt_keypoint_vio.cpp:226-230`
- Bias vector (first 3) seeds VIO initial dynamic bias: `basalt_slam.cpp:213-214`
- For SITL: all zero. For real hardware: run `basalt_calibrate_imu`
