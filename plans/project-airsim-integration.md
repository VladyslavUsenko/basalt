# Preparing Basalt to Consume Project AirSim

Status: PLAN — awaiting review.

## 1. Motivation

The SITL stack has migrated from Cosys-AirSim to Project AirSim, and `djinn start sitl2 basalt` now runs the Basalt visual-inertial odometry backend against a simulator whose inertial sensor is configured in different units, whose scene clock is a different object, and whose ROS bridge stamps messages by a different rule. The Basalt calibration at `ros_ws/src/slam/ext/basalt/data/sitl_calib.json` was written for the previous stack and the two context documents that justify it, [`../context/imu_noise_parameters.md`](../context/imu_noise_parameters.md) and [`../context/airsim_camera_extrinsics.md`](../context/airsim_camera_extrinsics.md), cite files that describe a simulator no longer in the loop.

Three questions had to be answered before the calibration could be trusted. Whether the noise parameters written into the Project AirSim robot configuration mean what the Basalt fields mean. Whether the coordinate frame reaching Basalt is still the one the extrinsics were derived in. And whether the sample rate and timestamps are now good enough for the inertial term to carry weight, given that the previous stack delivered a quarter of its samples and the rest were discarded as duplicates.

All three are now answered. The extrinsics require no change at all, and the sample rate is now adequate because the duplicate-stamp defect that halved the previous stack's inertial information is eliminated rather than merely reduced. Three of the four noise values in the calibration file are wrong and have been wrong since before the migration, two of them by simple arithmetic. The third, `gyro_bias_std`, is wrong for a second and larger reason that has nothing to do with either simulator, which is the finding this investigation did not set out to make and which is the substance of §2.4.

## 2. What the investigation established

### 2.1 Project AirSim reads SI directly, so the conversions vanish

Cosys-AirSim parses `settings.json` in engineering units and converts at load time, at `sitl_ws/src/ue5/AirSim/AirLib/include/sensors/imu/ImuSimpleParams.hpp:51-72`, dividing angular random walk by the root of 3600 and scaling degrees to radians, dividing gyroscope bias stability by 3600 likewise, multiplying velocity random walk by gravity over a thousand for milli-g, and multiplying accelerometer bias stability by 1e-6 and gravity for micro-g. Every conversion formula in the noise document existed to undo that.

Project AirSim performs none of it. `Imu::Loader::LoadAccelerometerSettings` at `sitl_ws/src/ue5/ProjectAirSim/core_sim/src/sensors/imu.cpp:294-316` and `LoadGyroSettings` at `:318-337` read each JSON key straight into a struct member through `JsonUtils::GetNumber<float>` with no arithmetic between file and member. The struct defaults at `core_sim/include/core_sim/sensors/imu.hpp:33-52` carry the conversion arithmetic as compile-time expressions, `0.24f * gravity / 1.0E3f` for the accelerometer and `0.3 / sqrt(3600.f) * M_PI / 180` for the gyroscope, which is the same MPU-6000 default figure Cosys-AirSim hard-codes. The units in the configuration file are therefore rad/s/√Hz, m/s²/√Hz, rad/s and m/s² respectively, and the mapping onto the two Basalt white-noise fields is the identity.

The user's hypothesis that no explicit conversion is needed is confirmed, and the reason is recorded so that the identity is not mistaken for an oversight.

The noise injection is also identical between the two. `ApplyNoiseModel` at `imu.cpp:209-236` forms `velocity_random_walk / sqrt_dt` and `angle_random_walk / sqrt_dt`, and division by the root of the timestep is the defining property of a continuous-time power spectral density, which is exactly what Basalt's `gyro_noise_std` and `accel_noise_std` are.

### 2.2 The bias process is the same pure random walk, so the divisor is still √τ

`imu.cpp:311-313` and `:333-334` precompute `bias_stability / sqrt(tau)` at load time, and `:222-224` and `:234-235` propagate the bias as `bias += gauss.next() * norm * sqrt_dt`. There is no mean-reversion term of the form −b/τ anywhere in the update, so the process is a plain Wiener process and matches Basalt's rate random walk with no approximation. The conversion is

    bias_std = bias-stability / √tau

with no factor of √2. This reproduces the correction already recorded for Cosys-AirSim on 2026-07-31 and confirms it on the new path.

Three properties of the loader deserve recording because none is visible from the configuration file. The two normalisation coefficients are declared without in-class initialisers at `imu.cpp:95` and assigned only inside the two loaders, which `LoadImuSettings` at `:270-292` skips entirely when the corresponding JSON object is absent, so a configuration omitting either block reads an uninitialised float on the first noise update. `turn-on-bias` seeds the bias state directly and is therefore a deterministic offset rather than a distribution. And the `accelerometer.gravity` key is loadable at `:296-297` but is never read again, since the specific force uses `env_info.gravity` instead at `:187-190`.

There is also no `GenerateNoise` flag. `ApplyNoiseModel` is called unconditionally at `imu.cpp:195`, so noise is always on and the only way to obtain a clean IMU is to zero all four parameters.

### 2.3 The numbers agree exactly, which is what makes the two simulators comparable

Applying the Cosys-AirSim conversions to the live `settings.json` and reading the Project AirSim configuration directly give the same four Basalt values to every digit carried.

| Basalt field | From Project AirSim, directly | From Cosys-AirSim, converted | Value |
|---|---|---|---|
| `gyro_noise_std` | `angle-random-walk` 5.817764e-4 | `AngularRandomWalk` 2.0 °/√hr | 5.8178e-4 rad/s/√Hz |
| `accel_noise_std` | `velocity-random-walk` 1.176798e-2 | `VelocityRandomWalk` 1.2 mg | 1.1768e-2 m/s²/√Hz |
| `gyro_bias_std` | 9.696274e-5 / √300 | 20.0 °/hr × 4.8481e-6 / √300 | 5.5981e-6 rad/s/√s |
| `accel_bias_std` | 4.903325e-3 / √600 | 500 μg × 9.80665e-6 / √600 | 2.0018e-4 m/s²/√s |

The identity is not a coincidence. Project AirSim's own example configurations reproduce the Cosys-AirSim defaults under the same conversions, 0.3 °/√hr giving 8.726646e-5 against an example 8.72644e-5 and 36 μg giving 3.530394e-4 against 3.53e-4, and the values in `robot_ardu_copter.jsonc:345-366` were obtained by applying those conversions to the live Cosys settings. Both simulators model the same MPU-6000 to within float precision, which is the property that makes a side-by-side comparison of the SLAM stack meaningful.

### 2.4 Basalt does not subscribe to the simulator, and the difference matters once

This is the finding the investigation did not set out to make. `djinn:409` passes `imu-topic:=/ap/imu/experimental/data`, an ArduPilot topic. The simulated inertial sample is serialised into the ArduPilot JSON packet at `sitl_ws/src/ue5/ProjectAirSim/vehicle_apis/multirotor_api/src/arducopter_api.cpp:705-714`, parsed at `ardu_ws/src/ardupilot/libraries/SITL/SIM_JSON.cpp:383-384`, and reaches `sitl->state.xAccel` through `Aircraft::fill_fdm` at `SIM_Aircraft.cpp:410-415`. ArduPilot's own simulated inertial backend then reads that state and adds three things to it.

It adds white noise. `AP_InertialSensor_SITL.cpp:120` adds one uniform draw of amplitude 0.01 m/s² per accelerometer sample and `:230-232` and `:247-249` add two independent draws of amplitude `radians(0.04)` per gyroscope sample, the second because the branch at `:245` is taken precisely when no vibration is configured. Neither amplitude is parameterised in the motors-off case; `SIM_ACC1_RND` and `SIM_GYR1_RND` default to zero and feed only the vibration terms, which are inert because `SIM_VIB_FREQ` and `SIM_VIB_MOT_MAX` default to zero. Since `rand_float` is uniform on [−1, 1) per `libraries/AP_Math/AP_Math.cpp:361-368`, the per-sample deviations at 1000 Hz are 5.7735e-3 m/s² and 5.7002e-4 rad/s, giving continuous densities of 1.8257e-4 and 1.8026e-5. Added in quadrature these raise `accel_noise_std` by 1.0001 and `gyro_noise_std` by 1.0005, so they are negligible and no calibration change follows.

It filters. `Tools/autotest/default_params/airsim-quadX.parm` sets `INS_ACCEL_FILTER` and `INS_GYRO_FILTER` to 50, against a hard-coded default of 20, so a single-pole low pass sits between the 1000 Hz generation and `AP::ins().get_accel()`. This is a trap rather than a correction. The visible per-sample scatter falls to σ_c·√((π/2)·50), which is 0.104 m/s², against the 0.152 that Basalt assumes when it forms σ_d = σ_c·√167 at `thirdparty/basalt-headers/include/basalt/calibration/calibration.hpp:147-161`. Anyone measuring the scatter directly and back-solving will conclude the calibration is 1.46 times too large. It is not. A unity-gain low pass leaves the low-frequency density untouched, and preintegration integrates the noise over intervals at least an order of magnitude longer than the filter's 3.2 ms correlation time, so the accumulated covariance is the one the unfiltered density predicts. The filter must not be compensated for.

And it drifts. `AP_InertialSensor_SITL::gyro_drift()` at `:401-413` adds a deterministic triangular ramp, identical on all three axes, of `SIM_DRIFT_SPEED` degrees per second per minute reversing every `SIM_DRIFT_TIME` minutes. The defaults at `libraries/SITL/SITL.cpp:73,77` are 0.05 and 5 and no parameter file in this stack overrides them, giving a ramp rate of 1.4544e-5 rad/s per second and a peak of 4.3633e-3 rad/s, which is 0.25 °/s. Basalt penalises the difference between consecutive gyroscope bias states by `gyro_bias_std·√dt` at `include/basalt/linearization/imu_block.hpp:73-85`, so the comparison is between a linear ramp and a square-root envelope.

| Window | ArduPilot ramp | 1σ permitted by 5.5981e-6 | Ratio |
|---|---|---|---|
| 0.1 s | 1.454e-6 | 1.770e-6 | 0.82 |
| 1 s | 1.454e-5 | 5.598e-6 | 2.60 |
| 10 s | 1.454e-4 | 1.770e-5 | 8.22 |
| 60 s | 8.727e-4 | 4.336e-5 | 20.12 |

The crossover is at 0.148 s and the ratio grows without bound thereafter. The estimator does not diverge, because the inertial residuals themselves push the bias state, but the prior fights a real and monotonic drift, a lagging gyroscope bias tilts the estimated attitude, and an attitude error θ leaks a horizontal specific force of roughly 9.81·sin θ that double-integrates into position. There is no accelerometer counterpart, since `generate_accel` adds only the constant `SIM_ACC1_BIAS`, which defaults to zero.

### 2.5 The frame is unchanged, so the extrinsics need no edit

Project AirSim is NED throughout. `Imu::Impl::Update` at `core_sim/src/sensors/imu.cpp:187-190` forms the specific force as `TransformVectorToBodyFrame(accels.linear − env_info.gravity, pose.orientation)`, and `Environment::SetPosition` at `core_sim/include/core_sim/environment.hpp:44-45` sets gravity to `(0, 0, +GetGravity(altitude))`, downward positive, so a stationary vehicle reports approximately `(0, 0, −9.805)`. The angular velocity is taken from `kinematics.twist.angular` at `:185` with no transform, correctly, because that member is body-frame, established by `physics/src/fast_physics.cpp:531-535` solving Euler's rigid-body equation with a body-frame inertia tensor. `arducopter_api.cpp:705-714` serialises both without conversion, ArduPilot reads them as its own front-right-down body vectors, and `AP_DDS_Client::update_topic(sensor_msgs_msg_Imu&)` at `libraries/AP_DDS/AP_DDS_Client.cpp:668-701` stamps `BASE_LINK_NED_FRAME_ID` and copies them verbatim.

The camera pose is likewise the same quantity in the same frame. `JsonUtils::GetTransform` at `core_sim/src/json_utils.cpp:101-128` reads `xyz` as a translation and passes `rpy-deg` through `TransformUtils::ToQuaternion`, whose implementation at `core_sim/src/transforms/transform_utils.cpp:21-51` is documented and written as the z-y-x Tait-Bryan composition R_z(ψ)·R_y(θ)·R_x(φ), exactly the convention the extrinsics derivation assumes. The vector order is roll, pitch, yaw, so `"rpy-deg": "0 -45 0"` is the `Pitch: -45` of the Cosys settings.

One premise that was previously presumed is now established. The IMU parses no `origin`. No call to `JsonUtils::GetTransform` appears anywhere in `imu.cpp`, unlike `camera.cpp:1462-1465`, `distance_sensor.cpp:251`, `magnetometer.cpp:294`, `rotor.cpp:398` and `wheel.cpp:409`, and `Update` applies no lever-arm or centripetal correction. The simulated IMU is unconditionally at the parent link's origin, which is `Frame`, the body origin. The camera-to-IMU extrinsic is therefore exactly the camera-to-body extrinsic with no residual translation, which is the assumption `T_imu_cam` has always rested on.

The conclusion is that no code change of any kind is required to put the IMU in the frame Basalt expects. The migration is frame-neutral on this topic.

The one path on which it would not be is the simulator's own inertial topic. `CreateImuPublisher` at `ros/projectairsim_ros2_cpp/src/projectairsim_ros2_cpp_node.cpp:397-416` passes both vectors through `ToRosVector3`, which at `include/projectairsim_ros2_cpp/ros2_conversion_utils.hpp:241-254` negates the second and third components, a 180° rotation about x that yields forward-left-up rather than a REP 103 east-north-up frame. §4.3 records the extrinsic that path would require.

### 2.6 The duplicate-stamp defect is eliminated, and 167 Hz is the right figure

The Cosys stack delivered 15.5 distinct inertial stamps per second against a 162 Hz message rate, because ArduPilot copies the last received `/clock` value verbatim into every `/ap/*` header and Cosys-AirSim published `/clock` at roughly 14 Hz from a callback starved by blocking msgpack round trips. Three of every four messages carried a repeated stamp and were dropped by the de-duplication filter at `ros_ws/src/slam/src/basalt/node.cpp:202-205`.

Project AirSim publishes `/clock` from a dedicated timer and `ros_ws/src/controllers/launch/sitl2.launch.py:95` sets `publish_clock_period_sec` to 0.005, a 200 Hz clock. The inertial gate is `AP_DDS_DELAY_IMU_TOPIC_MS` of 5 at `libraries/AP_DDS/AP_DDS_config.h:33-34`, tested as a strict inequality at `AP_DDS_Client.cpp:1930-1935`, giving a 6 ms period and a 166.7 Hz ceiling. Because the clock period is strictly shorter than the inertial period, at least one tick falls between any two consecutive samples, no two can share a stamp, and the distinct-sample rate equals the message rate. The 167 Hz the user reports is consistent with the gate; 162 Hz was measured on 2026-08-03 and the difference mis-scales σ_d by 1.5 percent, which is immaterial.

One property of `imu_update_rate` is worth stating because it is easy to get wrong. The gate is evaluated against `AP_HAL::millis()`, which under SITL is the stopped clock driven by the simulator's packet timestamps, so 167 Hz is a rate in simulation time. `imu_update_rate` lives in the same time domain as the header stamps, which under `use_sim_time` is also simulation time, so the figure is invariant to the ratio of `step-ns` to `real-time-update-rate` that scales the simulation against the wall clock. A stack running at one third real time still requires 167.

### 2.7 A discrepancy found in passing, and flagged rather than acted upon

`sitl_ws/config/airsim2/scene_ardu_copter.jsonc` sets `real-time-update-rate` to 9000000 against a `step-ns` of 3000000, which by `sitl_ws/src/ue5/ProjectAirSim/docs/config_scene.md:101-105` runs the simulation at one third of real time. [`/ws/context/sitl-time-synchronisation.md`](../../../../../context/sitl-time-synchronisation.md) records the value as 3000000, that is real time. One of the two is stale and I have not changed either, because the choice is a performance decision only the author can confirm.

Nothing in this plan depends on the resolution. Every noise figure is expressed in simulation time and the IMU timestep is `step-ns` regardless. The one place it shows is the effective stamp quantum, which is the coarser of the clock period and the simulation step, so 5 ms at real time and 3 ms at one third real time. Any wall-clock rate measurement must be scaled by the same factor before it is compared against 167 Hz.

## 3. The changes

### 3.1 Correct three noise values in the Basalt calibration

File `ros_ws/src/slam/ext/basalt/data/sitl_calib.json`, lines 109-129.

```json
        "imu_update_rate": 167,
        "accel_noise_std": [
            1.1768e-2,
            1.1768e-2,
            1.1768e-2
        ],
        "gyro_noise_std": [
            5.8178e-4,
            5.8178e-4,
            5.8178e-4
        ],
        "accel_bias_std": [
            2.0018e-4,
            2.0018e-4,
            2.0018e-4
        ],
        "gyro_bias_std": [
            5.5981e-6,
            5.5981e-6,
            5.5981e-6
        ],
```

Three values change and two do not. `accel_noise_std` falls from 0.020, which was produced by the generic ÷60 formula that neither simulator implements, to the configured density. `accel_bias_std` falls from 2.83e-4 and `gyro_bias_std` from 7.92e-6, both of which carry the spurious √2 of the first-order Gauss-Markov attribution that §2.2 disproves. `gyro_noise_std` is already correct to four significant figures and gains only precision. `imu_update_rate` is already 167 and is shown for completeness.

The `gyro_bias_std` entry of 5.5981e-6 is correct only in conjunction with §3.2. If §3.2 is rejected, this entry must instead be 1.5586e-5, the quadrature sum of 5.5981e-6 with the 1.4544e-5 that makes the ArduPilot ramp a one-sigma event over a one second window. The two changes are a package and must not be applied singly.

Need and significance. All three current values overstate the noise, which is the conservative direction, so the present configuration under-trusts the inertial term rather than diverging on it. Correcting them restores the inertial factors to the weight the sensor model justifies, which matters most on the accelerometer, where the error is a factor of 1.70.

Backwards-compatibility impact. This is the one change with a genuine consumer conflict, and it is discussed in §3.4 rather than here, because the resolution is structural.

### 3.2 Remove ArduPilot's deterministic gyroscope drift

File `sitl_ws/config/airsim2/project-airsim-quad.parm`, appended.

```
# ArduPilot's simulated inertial backend adds a deterministic triangular gyro
# drift of SIM_DRIFT_SPEED deg/s per minute, identical on all three axes,
# reversing every SIM_DRIFT_TIME minutes, at
# libraries/AP_InertialSensor/AP_InertialSensor_SITL.cpp:401-413. At the 0.05
# default it exceeds the bias excursion that the Project AirSim gyroscope
# random walk permits by a factor of 20 over one minute, so Basalt's bias prior
# fights a drift no simulator setting describes. Zeroing it leaves exactly one
# bias process in the stack, the one robot_ardu_copter.jsonc configures and
# sitl_calib.json models. See ros_ws/src/slam/ext/basalt/context/
# imu_noise_parameters.md Part 7.3.
SIM_DRIFT_SPEED 0
```

Mathematical background. `gyro_drift()` returns zero when either `SIM_DRIFT_SPEED` or `SIM_DRIFT_TIME` is zero, so a single parameter suffices and `SIM_DRIFT_TIME` is left alone. With the drift removed, the only gyroscope bias process in the stack is the Wiener process of `imu.cpp:234-235`, whose continuous-time coefficient is `bias-stability / √tau` by construction, and the calibration entry of §3.1 is then exact rather than fitted.

Why this rather than inflating the prior. Inflating `gyro_bias_std` models a deterministic ramp as a random walk it is not, and requires an arbitrary choice of the window over which the two are matched; a one second match gives 1.4544e-5, a ten second match gives 4.599e-5, and no principle selects between them. Removing the drift eliminates the mis-specification instead of accommodating it, and leaves the calibration derivable from the robot configuration alone, which is the property that makes it maintainable.

Why the loss of realism is acceptable. The purpose of this simulator configuration is a known inertial ground truth against which the estimator can be judged. A second, undocumented bias process defeats that, and a real device's bias behaviour is in any case better represented by raising `bias-stability` in the robot configuration, where it is visible, than by an ArduPilot default that is invisible from the SLAM side. Should a drifting bias be wanted deliberately, the correct route is to restore `SIM_DRIFT_SPEED` and set `gyro_bias_std` to 1.5586e-5 in the same commit.

Backwards-compatibility impact. `project-airsim-quad.parm` is referenced only by `ros_ws/src/controllers/launch/sitl2.launch.py:29`, as the default of the `ardupilot-params` launch argument, and is appended last to the SITL defaults list. It is not read by `sitl.launch.py`, so `djinn start sitl` is untouched. Within `sitl2`, removing a small gyroscope drift makes ArduPilot's own EKF task marginally easier and cannot destabilise a configuration that tolerates the drift today.

### 3.3 Correct the image timestamp arithmetic

File `ros_ws/src/slam/src/basalt/node.cpp`, lines 173-175.

```cpp
    int sec = msg->header.stamp.sec;
    int nsec = msg->header.stamp.nanosec;
    long timestamp = sec * 1e9 + nsec;
```

becomes

```cpp
    long timestamp = msg->header.stamp.sec * 1000000000LL +
                     msg->header.stamp.nanosec;
```

Need. The multiplication promotes to `double`. With `sec` around 1.7855e9 the product is roughly 1.7855e18, above the 2^53 threshold at which a double can no longer represent consecutive integers, and in that binade the representable spacing is 256, so image timestamps are quantised to 256 ns. The inertial path at `ros_ws/src/slam/src/slam.cpp:26-27` already uses integer arithmetic, so the two sensor paths disagree on how a timestamp is formed. The `int nsec` declaration is a second latent defect, since `stamp.nanosec` is `uint32_t` and values above 2^31 would overflow to negative, although the field is bounded by 1e9 in practice.

Significance. A 256 ns error is negligible beside a 6 ms inertial interval and this is not a correctness fix for the present stack. It is worth doing now because the migration has made the camera stamp meaningful: under Project AirSim the image carries the simulator's own per-sample `time_stamp` exact to the nanosecond, following the bridge patch recorded in [`/ws/context/sitl-time-synchronisation.md`](../../../../../context/sitl-time-synchronisation.md) under defect G12, so the quantisation now discards real precision rather than noise.

Backwards-compatibility impact. None. The function is `BasaltSLAMNode::GrabImage`, the value is consumed only by the `Frame` constructor two lines below, and the new expression is exactly the old one evaluated without floating-point rounding. It cannot change behaviour for any caller other than by removing the rounding.

### 3.4 Separate the two simulators' calibrations

`djinn:366` and `djinn:409` both pass `ros_ws/src/slam/ext/basalt/data/sitl_calib.json`, so the Cosys-AirSim and Project AirSim branches read one file. The four noise values are correct for both, since §2.3 establishes they are numerically identical. `imu_update_rate` is not. The file already carries 167, which is correct for Project AirSim and over-declares the Cosys rate by a factor of 10.8, inflating that path's per-sample noise by √10.8, which is 3.28.

This violates the rule that no change may alter the behaviour of a caller that did not ask for it, and the workspace keeps both simulators deliberately so that the same SLAM stack can be compared on identical inputs. A comparison run against a mis-scaled calibration is not a comparison.

The proposal is to add `ros_ws/src/slam/ext/basalt/data/sitl2_calib.json`, a copy of `sitl_calib.json` carrying the §3.1 values with `imu_update_rate` 167, and to restore `sitl_calib.json` to `imu_update_rate` 15.5 while taking the three corrected noise values, which are right for both. `djinn:409`, the `sitl2` branch, then reads the new file.

```bash
                    calibration-file-path:=/ws/ros_ws/src/slam/ext/basalt/data/sitl2_calib.json \
```

`djinn:366`, the `sitl` branch, is left untouched and continues to read `sitl_calib.json`.

Backwards-compatibility impact. This is the change that restores backwards compatibility rather than threatening it. `djinn start sitl basalt` returns to a calibration correct for the stack it runs, which it has not had since `imu_update_rate` was raised to 167. `djinn start sitl2 basalt` gains a file it alone owns, so future Project AirSim tuning cannot silently alter the Cosys baseline. The cost is one duplicated file whose four noise rows must be kept in step, which is acceptable because those rows change only when the robot configuration changes.

The alternative of keeping one file was considered and is rejected on the grounds above. It is recorded so that it is not reopened: it is cheaper by one file and invalidates every Cosys comparison, which is the entire reason the second stack is retained.

## 4. Deliberately not done

### 4.1 Compensating for the 50 Hz inertial filter

`INS_ACCEL_FILTER` and `INS_GYRO_FILTER` at 50 Hz reduce the visible per-sample scatter to 0.104 m/s² against the 0.152 the calibration implies. The temptation is to lower `accel_noise_std` to 8.07e-3 to match. That would be wrong, for the reason given in §2.4: a unity-gain low pass leaves the low-frequency density untouched, and preintegration integrates over intervals long compared with the filter's 3.2 ms correlation time, so the accumulated covariance is already correct. The temptation is recorded because measuring the scatter and back-solving is the obvious thing to do and gives the wrong answer.

### 4.2 Setting `cam_time_offset_ns`

Inertial stamps are quantised to the `/clock` tick while camera stamps are exact, so the inertial stream appears systematically early by half a quantum, partially cancelled by the filter's 3.2 ms group delay, leaving a residual of order a millisecond. `cam_time_offset_ns` cannot absorb it: the line that applies it is commented out at `ros_ws/src/slam/ext/basalt/src/vi_estimator/sqrt_keypoint_vio.cpp:231` and identically at `sqrt_keypoint_vo.cpp:178`, so the field is honoured only by the offline calibration tools. Setting it in the calibration file would have no effect on a live run and would mislead a later reader into believing the offset had been handled.

### 4.3 Moving Basalt onto the simulator's own inertial topic

The durable fix for both the quantisation of §4.2 and the second bias process of §3.2 is to stop consuming ArduPilot's inertial output and subscribe to `.../sensors/IMU1/imu` instead, which would give per-sample stamps exact to the nanosecond, 333 Hz instead of 167, and the configured noise model with nothing added. It is not undertaken here, consistent with the non-goal recorded in [`/ws/context/sitl-time-synchronisation.md`](../../../../../context/sitl-time-synchronisation.md), and because it trades a stream the flight controller actually fuses for one it does not, which changes what a comparison against ArduPilot's own state estimate means.

The extrinsic it would require is recorded so that the move can be made without re-deriving it. The bridge publishes through `ToRosVector3` at `ros/projectairsim_ros2_cpp/include/projectairsim_ros2_cpp/ros2_conversion_utils.hpp:241-254`, which negates the second and third components, a 180° rotation about x taking NED to forward-left-up. `T_imu_cam` is re-expressed by left-multiplying by `R_x(180°) = diag(1, −1, −1)`.

```json
{
    "px":  0.5,   "py":  0.0,   "pz": -0.1,
    "qx": -0.65328, "qy": 0.65328, "qz": -0.27060, "qw": 0.27060
}
```

The determinant is 1 and the quaternion norm is 1.00000000. The physical check is that the optical axis, the third column of the rotation, becomes `(0.70711, 0, −0.70711)`, which in a z-up frame is forward and 45° down, the same physical direction as the NED result. Note also that the frame is forward-left-up and not REP 103 east-north-up, so a downstream consumer expecting ENU would be wrong by a yaw as well.

### 4.4 Re-deriving the extrinsics or the intrinsics

Neither changes. §2.5 establishes that the pose, the frame and the Euler convention are the same, and `robot_ardu_copter.jsonc:324-336` sets 640 by 480 at 90 degrees, the same ideal pinhole for which `fx = fy = 320`, `cx = 320` and `cy = 240` were verified on 2026-07-31. `sitl_calib.json:3-30` is correct as it stands.

## 5. Blast radius

| File and line | Reference | Treatment |
|---|---|---|
| `ros_ws/src/slam/ext/basalt/data/sitl_calib.json:109-129` | the four noise values and the rate | three values corrected, rate restored to 15.5 for the Cosys path, §3.1 and §3.4 |
| `ros_ws/src/slam/ext/basalt/data/sitl2_calib.json` | new file | Project AirSim calibration, §3.4 |
| `djinn:366` | `sitl` branch `calibration-file-path` | untouched, continues to read `sitl_calib.json` |
| `djinn:409` | `sitl2` branch `calibration-file-path` | repointed at `sitl2_calib.json`, §3.4 |
| `ros_ws/src/slam/ext/basalt/data/sitl_config_vo.json` | solver and optical-flow settings | untouched, carries no inertial parameter |
| `sitl_ws/config/airsim2/project-airsim-quad.parm` | ArduPilot tuning, `sitl2` only | one parameter appended, §3.2 |
| `sitl_ws/config/airsim2/robot_ardu_copter.jsonc:345-366` | the IMU noise block | untouched, verified correct in §2.3 |
| `sitl_ws/config/airsim2/robot_ardu_copter.jsonc:313-344` | the camera origin and capture settings | untouched, verified correct in §2.5 and §4.4 |
| `ros_ws/src/slam/src/basalt/node.cpp:173-175` | image timestamp arithmetic | corrected, §3.3 |
| `ros_ws/src/slam/src/basalt/node.cpp:196-207` | inertial de-duplication filter | untouched, now drops nothing, §2.6 |
| `ros_ws/src/slam/src/slam.cpp:25-27` | inertial timestamp arithmetic | untouched, already correct and is the model for §3.3 |
| `ros_ws/src/slam/include/slam/slam.hpp:97-103` | `toBasaltImuData` | untouched, no frame conversion needed, §2.5 |
| `ros_ws/src/controllers/launch/sitl2.launch.py:95` | `publish_clock_period_sec` 0.005 | untouched, is what eliminates the duplication, §2.6 |
| `ros_ws/src/slam/orb_slam3/config/Monocular/sitl.yaml` | ORB-SLAM3 camera settings | untouched, this plan concerns Basalt only |

No C++ in `ros_ws` hardcodes a simulator topic or a frame identifier, so §3.3 is the only recompilation, and it is confined to the `slam` package.

## 6. Validation

None of these can run in the development container, which has no Docker daemon. They are authored here so that they can be executed against a running stack.

| # | Command or action | Assertion |
|---|---|---|
| V1 | `python3 -c` on the four conversions of §2.3 | reproduces 5.8178e-4, 1.1768e-2, 5.5981e-6 and 2.0018e-4. Run, passes. |
| V2 | `python3 -m json.tool` on both calibration files | valid JSON, and the two differ only in `imu_update_rate` |
| V3 | `ros2 topic hz /ap/imu/experimental/data` under `djinn start sitl2` | approximately 167 Hz in the stamp domain, scaled by the simulation speed factor if read on the wall clock |
| V4 | unique header stamps against message count, per the procedure in [`../context/ap_dds_imu_stream.md`](../context/ap_dds_imu_stream.md) | ratio 1.00, confirming §2.6 |
| V5 | histogram of consecutive stamp deltas | multiples of the effective quantum, centred on 6 ms |
| V6 | sample standard deviation of `linear_acceleration`, stationary and disarmed | approximately 0.104 m/s², the post-filter figure. Not 0.152, and §4.1 explains why that is not a discrepancy |
| V7 | mean `angular_velocity` over ten minutes, before and after §3.2 | a triangular ramp of peak 4.36e-3 rad/s and period ten minutes before, flat after |
| V8 | `djinn start sitl2 basalt` on a stationary vehicle, before and after §3.1 | position drift over sixty seconds does not increase; the inertial term is being trusted more, so any regression indicates the noise model is optimistic rather than conservative |
| V9 | `djinn start sitl basalt` after §3.4 | unchanged against the pre-migration baseline, confirming the Cosys path is restored |
| V10 | `ros2 topic hz` on the camera topic | measures what `capture-interval` 0.05 achieves, so that `sitl.yaml:28` `Camera.fps` can be corrected from evidence. Outstanding from the workspace plan, repeated here because it bears on Basalt too |

## 7. Documentation already updated

The three context documents were corrected in place ahead of this plan, since they are the reference the changes are justified from.

- [`../context/imu_noise_parameters.md`](../context/imu_noise_parameters.md), retitled to Project AirSim, with the Cosys conversions retained in parallel because that stack is still runnable. New Project AirSim ground-truth table, identity mappings in Parts 1.1 and 1.2, section 2.1c on the loader and its three gotchas, the scene-tick sample rate and `min_sample_time` floor in Part 3, the absence of `GenerateNoise` in Part 4, the Project AirSim value table and the corrected discrepancy table in Part 5, and a new Part 7 deriving everything in §2.4 above.
- [`../context/airsim_camera_extrinsics.md`](../context/airsim_camera_extrinsics.md), with the `origin` and `rpy-deg` sources, the z-y-x convention verified against `transform_utils.cpp`, the 89.9° pitch clamp, the stale `/ws/envs/...` path corrected, and a 2026-08-17 verification section covering the IMU-at-body-origin proof and the forward-left-up extrinsic of §4.3.
- [`../context/ap_dds_imu_stream.md`](../context/ap_dds_imu_stream.md), with a Project AirSim section recording what is unchanged, the elimination of the duplicate-stamp defect, the residual quantisation, the second bias process, and the outstanding measurements.
- [`../context/README.md`](../context/README.md), index rows refreshed and the quick-reference split into Project AirSim and Cosys-AirSim conversion blocks with a new block on what stands between the simulator and Basalt.

[`../context/imu_static_calibration.md`](../context/imu_static_calibration.md) was checked and needs no edit. `calib_accel_bias` and `calib_gyro_bias` remain all zeros, correctly, because Project AirSim's `turn-on-bias` is zero and it models no scale or misalignment error at all.
