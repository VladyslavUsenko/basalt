# AP_DDS IMU Stream, Frame, Rate and Timestamps

This document records what the Basalt backend actually receives on the inertial topic under Project AirSim, as distinct from what the robot configuration nominally describes. The distinction matters because the IMU that Basalt consumes does not come from the Project AirSim inertial sensor at all, it comes from ArduPilot by way of the AP_DDS bridge, so the transport, the frame identifier, the sample rate and the timestamping are all ArduPilot's while only the noise characteristics originate in the simulator.

The stack under discussion is the one launched by `djinn start sitl2`, in which Project AirSim runs the physics and rendering, `ros_ws/src/controllers/launch/sitl2.launch.py` brings up the Project AirSim ROS 2 bridge alongside ArduPilot SITL, and the SLAM node subscribes to `/ap/imu/experimental/data` for inertial data and to `/airsim_node/Copter/front_center_Scene/image` for imagery.

## Provenance of the stream

`src/basalt/node.cpp:112-123` subscribes to the topic named by the `imu_topic_name` parameter, and the comment on those lines records the topic in use as `/ap/imu/experimental/data`, published by AP_DDS. The subscription uses `rclcpp::SensorDataQoS()` because AP_DDS publishes BEST_EFFORT and a RELIABLE subscriber would never match, in which case the callback would silently never fire.

The path from the simulated sensor to that topic runs as follows. `Imu::Impl::Update` at `sitl_ws/src/ue5/ProjectAirSim/core_sim/src/sensors/imu.cpp:174-207` produces a noisy body-frame sample once per scene tick, `ArduCopterApi::SendSensorData` at `vehicle_apis/multirotor_api/src/arducopter_api.cpp:705-714` serialises it into the ArduPilot SITL JSON packet without conversion, `libraries/SITL/SIM_JSON.cpp:383-384` parses it into `accel_body` and `gyro` on the flight dynamics model, `AP_InertialSensor_SITL` adds ArduPilot's own noise and filtering, and `AP_DDS_Client::update_topic(sensor_msgs_msg_Imu&)` at `libraries/AP_DDS/AP_DDS_Client.cpp:668-701` copies `AP::ins().get_accel()` and `get_gyro()` verbatim into the message and stamps the header with `BASE_LINK_NED_FRAME_ID`.

## Frame convention, confirmed NED

A captured message from a stationary, disarmed vehicle carries the following, abbreviated from a `ros2 topic echo` of the full payload.

```
header:
  stamp:
    sec: 1786985213
    nanosec: 354770000
  frame_id: base_link_ned
orientation:
  x: 0.9999442100524902
  y: -0.001409408519975841
  z: -0.0002012890763580799
  w: -0.010468179360032082
orientation_covariance: [0.0 × 9]
angular_velocity:
  x: 0.008249937556684017
  y: 0.007056903094053268
  z: 0.004868494346737862
angular_velocity_covariance: [0.0 × 9]
linear_acceleration:
  x: -0.0610186830163002
  y: -0.010614397935569286
  z: -9.874547004699707
linear_acceleration_covariance: [0.0 × 9]
```

Two independent facts establish the convention. The frame identifier states `base_link_ned` explicitly, and the stationary specific-force measurement is approximately negative 9.81 on the z axis. A stationary accelerometer measures specific force, which is the negative of the gravity vector expressed in the body frame, so a reading of −9.81 on z places the z axis pointing down.

The IMU frame is therefore NED, meaning x forward, y right and z down. This is the same convention as the Project AirSim body frame, which is what [`airsim_camera_extrinsics.md`](airsim_camera_extrinsics.md) assumes when deriving `T_imu_cam`, so the extrinsic derivation rests on a premise confirmed at the consuming end rather than merely asserted at the producing end.

The NED reading is corroborated from the simulator side. `Imu::Impl::Update` at `core_sim/src/sensors/imu.cpp:187-190` forms the specific force as `TransformVectorToBodyFrame(accels.linear − env_info.gravity, pose.orientation)`, and `Environment::SetPosition` at `core_sim/include/core_sim/environment.hpp:44-45` sets `env_info.gravity` to `(0, 0, +GetGravity(altitude))`, downward positive in NED, so a stationary vehicle necessarily reports approximately `(0, 0, −9.805)`. The angular velocity is taken from `kinematics.twist.angular` at `:185` with no transform, which is correct because that member is body-frame, established by `fast_physics.cpp:531-535` solving Euler's rigid-body equation with a body-frame inertia tensor.

Across the captured window the accelerometer z component ranges over roughly −9.797 to −9.892 and the angular velocity components over roughly 1e-3 to 1e-2 rad/s, both consistent with the configured noise densities derived in [`imu_noise_parameters.md`](imu_noise_parameters.md).

Note that all three covariance arrays are identically zero. AP_DDS never populates them, and Basalt never reads them, since the estimator takes its noise model entirely from the calibration file. A consumer that trusted the covariance field would infer a perfect sensor.

## The orientation field, and why it is unused

The captured `orientation` quaternion is approximately x 0.99994, y −0.00141, z −0.00020, w −0.01047, which is a rotation of very nearly 180 degrees about the x axis. This is the expected relationship between a NED body frame and an ENU or FLU reference, and it is a useful corroboration of the NED reading above.

Basalt never sees it. `Imu::toBasaltImuData` in `include/slam/slam.hpp:97-103` populates only `t_ns`, `accel` and `gyro`. The orientation is parsed into the `Imu` object at `src/slam.cpp:43-49` and then discarded. No frame conversion of any kind is applied to the accelerometer or gyroscope vectors between the ROS message and Basalt, so whatever frame the topic publishes in is the frame in which `T_imu_cam` must be expressed. `mpBasaltToROSTransform` is set to identity at `src/basalt/node.cpp:46` and does not participate in the ingest path.

The consequence worth stating plainly is that no code change is required to put the inertial data into the frame Basalt expects. The stream is NED end to end and the calibration is expressed in NED, so the two already agree.

## Gravity handling in Basalt, and why NED does not break it

Basalt defines the world gravity vector at `include/basalt/utils/imu_types.h:62`.

```cpp
static const Eigen::Vector3d g(0, 0, -9.81);
```

The world frame is therefore z-up. A naive reading suggests that a z-down IMU would be catastrophic, since a stationary NED accelerometer reports −9.81 on z while a z-up world expects +9.81, which would appear to the estimator as a two-g downward acceleration. That reading is incorrect for this codebase, and the reason is worth recording because it is a trap.

`BasaltSLAM::InitialiseSlam` at `src/basalt/slam.cpp:48-55` calls `Controller::initialize` with `t_ns` zero, `T_w_i` identity and zero velocity. `src/controller.cpp:163-167` inspects exactly that combination.

```cpp
if (t_ns == 0 && T_w_i.log().norm() < 1e-9 && vel_w_i.norm() < 1e-9) {
    vio_estimator_->initialize(bg, ba);
} else {
    vio_estimator_->initialize(t_ns, T_w_i, vel_w_i, bg, ba);
}
```

The two-argument overload leaves the member `initialized` false, so the estimator later takes the deferred gravity-alignment branch at `src/vi_estimator/sqrt_keypoint_vio.cpp:234-249`.

```cpp
T_w_i_init.setQuaternion(Eigen::Quaternion<Scalar>::FromTwoVectors(
    imuData->accel, Vec3::UnitZ()));
```

The initial attitude is constructed to rotate the first measured specific-force vector onto world +Z, so the estimator adapts to whichever direction the IMU calls up. A NED IMU is handled correctly, and the accelerometer sign convention is not a defect in this stack.

One caveat follows from the degenerate geometry. A NED stationary accelerometer is very nearly antiparallel to +Z, and `Eigen::Quaternion::FromTwoVectors` on antiparallel inputs must choose a rotation axis arbitrarily from the orthogonal plane. Gravity alignment is exact regardless, but the initial heading of the world frame is arbitrary and not reproducible between runs. Yaw is an unobservable gauge freedom in monocular VIO in any case, so this affects the interpretation of logged trajectories rather than the estimate quality. It does mean that comparing a live trajectory against a fixed ground truth requires the SE(3) first-pose alignment described in [`gt_slam_alignment.md`](gt_slam_alignment.md).

Note also that the four-argument branch is dead for the current caller. Any future change that passes a non-identity `T_w_i` would suppress gravity alignment entirely and would then require the caller to supply a gravity-consistent attitude by hand.

## Timestamps, the mechanism that makes every stamp distinct

### The two sources ArduPilot can stamp from

Every `/ap/*` header is stamped by one function, `AP_DDS_Client::update_topic(builtin_interfaces_msg_Time&)` at `libraries/AP_DDS/AP_DDS_Client.cpp:214-232`, which chooses between two sources.

```cpp
void AP_DDS_Client::update_topic(builtin_interfaces_msg_Time& msg)
{
#if AP_DDS_CLOCK_SUB_ENABLED
    // use external /clock topic if available
    if (has_received_clock) {
        msg.sec = external_clock_time.sec;
        msg.nanosec = external_clock_time.nanosec;
        return;
    }
#endif
    uint64_t utc_usec;
    if (!AP::rtc().get_utc_usec(utc_usec)) {
        utc_usec = AP_HAL::micros64();
    }
    msg.sec = utc_usec / 1000000ULL;
    msg.nanosec = (utc_usec % 1000000ULL) * 1000UL;
}
```

The external branch is armed by `AP_DDS_CLOCK_SUB_ENABLED`, which `libraries/AP_DDS/AP_DDS_config.h:121-122` defines as true on SITL, and latched by `has_received_clock`, set once and never cleared in the `CLOCK_SUB` deserialisation case at `:917`. It copies the last value received on the ROS `/clock` topic, held constant between arrivals, so on this branch the stamp is quantised to the clock publication period.

The fallback branch reads the simulated real-time clock. `AP_RTC::get_utc_usec` at `libraries/AP_RTC/AP_RTC.cpp:106-113` returns `AP_HAL::micros64() + rtc_shift`, and under SITL with `synthetic_clock` enabled `micros64()` is the stopped clock driven by the simulator's packet timestamps rather than the wall clock. The stamp on this branch is therefore simulation time at microsecond resolution, offset by a constant UTC epoch captured when the RTC was first set, and it is strictly monotonic with no quantisation of its own.

### The publication gate that sets the period

Both branches are sampled at the same cadence. `AP_DDS_Client::write_imu_topic` is called only through the gate at `AP_DDS_Client.cpp:1930-1935`.

```cpp
if (cur_time_ms - last_imu_time_ms > DELAY_IMU_TOPIC_MS) {
    update_topic(imu_topic);
    last_imu_time_ms = cur_time_ms;
    write_imu_topic();
}
```

`AP_DDS_DELAY_IMU_TOPIC_MS` is 5 at `libraries/AP_DDS/AP_DDS_config.h:33-34` and the comparison is a strict inequality, so the shortest permitted interval is 6 ms and the ceiling is 166.7 Hz. The gate is evaluated against `AP_HAL::millis()`, which under SITL is the same stopped clock, so 167 Hz is a rate in simulation time and is invariant to the ratio of `step-ns` to `real-time-update-rate` that scales the simulation against the wall.

### What the capture shows

Sixty-nine consecutive stamps read from `/ap/imu/experimental/data` on a live session begin as follows and continue in the same pattern.

```
    sec: 1786985195
    nanosec: 864770000
    sec: 1786985195
    nanosec: 870770000
    sec: 1786985195
    nanosec: 876770000
    sec: 1786985195
    nanosec: 882770000
    ...
    sec: 1786985196
    nanosec: 272770000
```

Every one of the sixty-nine stamps is distinct, the sixty-eight intervals are all exactly 6,000,000 ns with no jitter whatever, the window spans 0.408 s and the implied rate is 166.667 Hz, which is the gate ceiling attained exactly. The sub-millisecond part is a constant 770 µs, so the residue modulo the 6 ms period is fixed at 2,770,000 ns.

That signature identifies the branch in use. The residues modulo 5 ms take five distinct values across the window, so the stamps are not quantised to a 200 Hz `/clock`, and the leading value of 1786985195 decodes as a UTC calendar instant contemporaneous with the capture, which is a wall-clock epoch rather than a simulation time. The capture was therefore stamped from the `AP::rtc()` fallback, meaning `has_received_clock` was false and no `/clock` message had reached ArduPilot's DDS client during that run. The exactly periodic 6 ms spacing at microsecond resolution is precisely what that branch predicts, since the stamp is simulation time read at the instant the gate fires.

### Why duplication cannot occur on either branch

On the RTC branch the argument is immediate. The stamp is a strictly increasing function of simulation time at microsecond resolution, and the gate guarantees at least 6 ms of simulation time between successive writes, so consecutive stamps differ by at least 6,000 µs and can never coincide.

On the `/clock` branch the argument rests on a rate comparison. `sitl2.launch.py:95` sets `publish_clock_period_sec` to 0.005, a 200 Hz clock, against the 6 ms inertial period. Because the clock quantum is strictly shorter than the inertial period, at least one clock tick necessarily falls between any two consecutive inertial samples, so the quantised values differ by at least one quantum and again can never coincide. The effective quantum is bounded by the coarser of the clock publication period and the simulation step, because `PublishClock` at `ros/projectairsim_ros2_cpp/src/projectairsim_ros2_cpp_node.cpp:1000-1014` returns the scene's simulation time, which advances only in `step-ns` increments. With `step-ns` of 3 ms and a clock timer of 5 ms of wall time the quantum is 5 ms when the simulation runs at real time and 3 ms whenever it runs slower than 0.6 of real time, and `scene_ardu_copter.jsonc` currently sets `real-time-update-rate` to 9000000 against a `step-ns` of 3000000, a ratio of one third real time, under which the quantum is the 3 ms step. Either value is comfortably below 6 ms, so the conclusion holds throughout the configured range.

Duplicate inertial stamps are therefore impossible in this stack irrespective of which branch is live, and the distinct-sample rate equals the message rate. `imu_update_rate` in the Basalt calibration must consequently be set to the message rate of 167 Hz, and because that rate lives in the same stamp time domain it is a simulation-time figure. A stack running at one third real time still requires 167, not 55.7.

### The de-duplication filter in the ingest node

The SLAM node nonetheless guards against duplicate and out-of-order stamps, and the guard is described here in full because it is the mechanism that would absorb any future regression in the timestamping above. `BasaltSLAMNode::GrabIMU` at `ros_ws/src/slam/src/basalt/node.cpp:190-211` compares each arriving stamp against the last accepted one.

```cpp
if (mpLastIMUTimestamp == -1) {
    mpLastIMUTimestamp = imuPtr->getTimestampNSec();
} else if (mpLastIMUTimestamp > imuPtr->getTimestampNSec()) {
    RCLCPP_DEBUG(this->get_logger(), "Received out of order IMU, dropping");
    return;
} else if (mpLastIMUTimestamp == imuPtr->getTimestampNSec()) {
    RCLCPP_DEBUG(this->get_logger(), "Received duplicate IMU, dropping");
    return;
}
mpLastIMUTimestamp = imuPtr->getTimestampNSec();
mpSlam->GrabIMU(imuPtr);
```

The first sample seeds the state and is forwarded, a strictly earlier stamp is discarded as out of order, an identical stamp is discarded as a duplicate, and anything strictly later is accepted and becomes the new reference. The intent is correct, since a zero time delta between consecutive samples would produce a degenerate preintegration interval, and a negative delta would produce a preintegration covariance that is not positive definite.

Under the mechanism established above the filter is inert. It drops nothing, because no two stamps ever coincide and the stream is monotonic by construction, so the samples reaching Basalt are exactly the samples published. Its cost is one integer comparison per message and it is worth keeping precisely because it converts a silent numerical failure into a debug-level log line should the stamp source ever change.

Two properties of the filter deserve recording against that eventuality. It is unconditional on the delta magnitude, so it accepts an arbitrarily small positive interval without complaint, and it retains the first message of any duplicated group rather than the last or an average, so a stream that did duplicate would be decimated by arrival order rather than properly resampled.

## The epoch mismatch between the inertial and visual streams

The two streams are not stamped from the same source, and a capture of the camera topic taken on the same stack makes the divergence explicit. Frames on `/airsim_node/Copter/front_center_Scene/image` carry `sec` values between 1399 and 1456, which is raw simulation time measured from scene start, while inertial samples on `/ap/imu/experimental/data` carry `sec` values around 1786985195, a UTC-derived epoch.

The reason is structural rather than accidental. The image stamp originates in the simulator payload, since `MakeHeader(rclcpp::Node&, const std::string&, const json&)` at `projectairsim_ros2_cpp_node.cpp:122-133` stamps from the message's own `time_stamp` field when the payload carries one, and Project AirSim's simulation clock starts at zero, because `SteppableClock` is declared at `core_sim/include/core_sim/clock.hpp:95-96` with a default `start` of 0, `ClockSettings` at `:193-198` exposes no key that could change it, and `scene.cpp:1081` constructs it as `std::make_shared<SteppableClock>(impl_.clock_settings_.step)` with the start defaulted. The inertial stamp, on the capture analysed above, originates in ArduPilot's RTC and therefore carries the wall-clock epoch that the RTC was seeded with.

If both captures come from one session the two streams differ by a constant offset of roughly 1.787e9 s, which Basalt cannot fuse. Its preintegration interval for a frame is the span between the frame stamp and the surrounding inertial stamps, so an offset of that magnitude places every image outside the inertial window entirely and the estimator receives no usable visual-inertial constraint at all. This is not a calibration matter and `cam_time_offset_ns` cannot absorb it, both because the offset is nine orders of magnitude beyond anything that field is intended for and because the line that would apply it is commented out at `src/vi_estimator/sqrt_keypoint_vio.cpp:231`, with the identical line commented out in the visual-only estimator at `sqrt_keypoint_vo.cpp:178`, so the field is honoured only by the offline calibration tools.

The remedy is to place both streams in simulation time by ensuring ArduPilot takes the `/clock` branch, which is what `sitl2.launch.py` already intends, since it publishes `/clock` at 200 Hz and passes `synthetic_clock` and `use_sim_time` to the ArduPilot launch. The capture shows that intent unrealised, so the connectivity of `/clock` into ArduPilot's DDS domain is the thing to verify on a running stack. The check is direct, namely that the `sec` field of an inertial header should read in the low thousands and should agree with `ros2 topic echo /clock`, not in the billions.

A second and much smaller timing residual survives even once both streams share an epoch. On the `/clock` branch an inertial sample taken at simulation time t is stamped with the most recent clock value ArduPilot has received, understating t by up to one quantum and on average by half of one, while the camera carries the simulator's own per-sample `time_stamp` exact to the nanosecond, so the inertial stream appears systematically early by half a quantum. Working the other way, the 50 Hz `INS_GYRO_FILTER` and `INS_ACCEL_FILTER` set by `Tools/autotest/default_params/airsim-quadX.parm` delay the inertial payload by roughly 1/(2π·50), which is 3.2 ms, so the two errors partially cancel and the residual is of order a millisecond. On the RTC branch the quantisation term vanishes and only the filter delay remains.

## Timestamp arithmetic in the ingest path

`src/slam.cpp:26-27` computes the IMU timestamp using integer arithmetic, which is correct.

```cpp
mpTimestamp = msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;
```

The image path does not match. `src/basalt/node.cpp:172-175` uses a floating-point literal.

```cpp
int sec = msg->header.stamp.sec;
int nsec = msg->header.stamp.nanosec;
long timestamp = sec * 1e9 + nsec;
```

The multiplication promotes to `double`. The severity depends on which epoch the stamp carries. With simulation-time stamps of order 1.4e3 s the product is about 1.4e12, well inside the 2^53 range in which a double represents every integer exactly, so the defect is currently latent on the image path. Should the image stamps ever carry a UTC-derived epoch, as the inertial stamps presently do, the product would be about 1.787e18, above the 2^53 threshold and in a binade whose representable spacing is 256 ns, quantising every image timestamp accordingly. The line should be brought into line with the IMU path regardless, since it is a real inconsistency between the two sensor paths and its correctness depends on a property of the data rather than of the code. Note additionally that `nsec` is declared `int` while `stamp.nanosec` is `uint32_t`, so values above 2^31 would overflow into negative, although the field is bounded by 1e9 in practice.

## The sample rate is what the calibration must declare

Basalt forms the per-sample noise standard deviation as σ_c·√rate at `thirdparty/basalt-headers/include/basalt/calibration/calibration.hpp:147-161`, so `imu_update_rate` must be the rate at which distinct samples arrive at the subscriber, in the time domain of the header stamps. The capture establishes that figure as 166.667 Hz exactly, and 167 is the value carried in `data/sitl_calib.json`. An earlier measurement on this stack reported 162 Hz, against which 167 mis-scales σ_d by √(167/162), which is 1.5 percent and immaterial.

A 167 Hz inertial stream is comfortable for visual-inertial odometry. Preintegration between successive camera frames spans many samples and attitude propagation over 6 ms intervals accumulates little error, which matters because gravity is roughly 9.81 m/s² and an attitude error θ leaks a horizontal specific-force error of approximately 9.81·sin θ into the estimate, which double integrates into position. That is the classical mechanism by which a stationary platform produces a diverging trajectory, and at this rate it is not the limiting one.

## What the simulator configuration does not describe

The noise on this topic is not entirely the simulator's. `AP_InertialSensor_SITL::generate_accel` and `generate_gyro` at `libraries/AP_InertialSensor/AP_InertialSensor_SITL.cpp:114-120` and `:218-249` add ArduPilot's own white noise on top of whatever the external simulator supplies, and `gyro_drift()` at `:401-413` adds a deterministic triangular ramp of 0.05 °/s per minute, identical on all three axes, reversing every five minutes at a peak of 0.25 °/s. The additive white noise is negligible, raising the densities by factors of 1.0001 and 1.0005. The drift is not, exceeding the one-sigma excursion that the simulator-derived `gyro_bias_std` permits by a factor of 2.6 over one second and 20 over one minute. The full derivation, the comparison table and the two available remedies are in [`imu_noise_parameters.md`](imu_noise_parameters.md) Part 7.

## Verification procedure

To confirm on a running stack that every stamp is distinct, count unique header stamps rather than messages.

```bash
ros2 topic echo /ap/imu/experimental/data --field header.stamp | \
  paste - - | sort -u | wc -l
```

Compare that against the raw message rate reported by `ros2 topic hz /ap/imu/experimental/data`. A ratio of one confirms the mechanism described above. A ratio materially above one would indicate that the stamp source has changed to something coarser than the inertial period, in which case the gate period and the clock quantum are the two quantities to re-measure.

To confirm that the inertial and visual streams share a time base, compare the `sec` field of the two topics against `/clock`.

```bash
ros2 topic echo /clock --field clock.sec --once
ros2 topic echo /ap/imu/experimental/data --field header.stamp.sec --once
ros2 topic echo /airsim_node/Copter/front_center_Scene/image --field header.stamp.sec --once
```

All three should agree to within a second. A nine-order-of-magnitude disagreement between the first two identifies the epoch mismatch described above.

## Measurements outstanding

None of these can run in the development container, which has no Docker daemon. They are what would convert the derived parts of this record into measured ones.

| # | Command | Assertion |
|---|---|---|
| I1 | `ros2 topic hz /ap/imu/experimental/data` | approximately 167 Hz in the stamp time domain, scaled by the simulation speed factor if measured on the wall clock |
| I2 | the unique-stamp count above, against `ros2 topic hz` | ratio 1.00, confirming every stamp is distinct |
| I3 | the epoch comparison above | all three topics agreeing, confirming `has_received_clock` is true and the streams share simulation time |
| I4 | sample standard deviation of `linear_acceleration` while stationary and disarmed | approximately 0.104 m/s², the post-filter figure of `imu_noise_parameters.md` Part 7.2, and not the 0.152 that the calibration implies |
| I5 | drift of the mean `angular_velocity` over ten minutes | a triangular ramp of peak 4.36e-3 rad/s and period ten minutes if `SIM_DRIFT_SPEED` is left at its default, flat if it is set to zero |

## Related context

- [`imu_noise_parameters.md`](imu_noise_parameters.md), the noise parameter mapping and the three effects ArduPilot adds to this stream.
- [`airsim_camera_extrinsics.md`](airsim_camera_extrinsics.md), the extrinsic derivation resting on the NED frame confirmed here, and the camera frame rate measured on the same stack.
- [`gt_slam_alignment.md`](gt_slam_alignment.md), the first-pose alignment needed because the initial heading is arbitrary.
- [`vio_localmapper_correction_loop.md`](vio_localmapper_correction_loop.md), why the live trajectory is never retro-corrected by the local mapper.
- [`/ws/context/sitl-time-synchronisation.md`](../../../../../context/sitl-time-synchronisation.md), the cross-workspace treatment of clock and stamp behaviour in the SITL stack.
