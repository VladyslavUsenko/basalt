# Project AirSim Camera Extrinsics → Basalt T_imu_cam

## Problem Statement

Project AirSim defines each camera by a position and an orientation in the NED body frame, written into the sensor entry of the robot configuration as a single `origin` object with two string-valued keys.

- **`xyz`** — three floats, the camera origin in the parent link's NED frame (metres)
- **`rpy-deg`** — three floats in the order roll, pitch, yaw, the camera frame orientation relative to the parent link (degrees)

Basalt stores `T_imu_cam` (= `T_i_c`) per camera: the SE3 that maps a point from the
**OpenCV camera frame** into the **IMU / body frame**:

    p_body = R_i_c · p_cam + t_i_c

This document derives the formula, applies it to the `front_center` camera, verifies the result numerically and against a live capture, and records the intrinsics, the achieved frame rate and the one alternative frame in which the extrinsic would have to be re-expressed.

---

## Coordinate Frames

| Symbol | Name | Convention |
|--------|------|------------|
| **B** | Body / IMU | NED: X forward, Y right, Z down |
| **CA** | Project AirSim camera | Same as B when roll=pitch=yaw=0 (NED) |
| **CCV** | Basalt / OpenCV camera | X right, Y down, Z forward (optical axis) |

The NED convention is not an assumption but a property of the codebase, corroborated at three independent points. `core_sim/include/core_sim/physics_common_types.hpp:201-202` declares the body forward vector as `(1, 0, 0)` and the upward vector as `(0, 0, -1)` with the comment `local NED frame unit vector`. The rotor `normal-vector` of `"0.0 0.0 -1.0"` is documented as straight up at `docs/config_robot.md:615`. And the topic Basalt consumes reports `frame_id: base_link_ned` with a stationary specific force of approximately (−0.06, −0.01, −9.87), which places z downward, as recorded in [`ap_dds_imu_stream.md`](ap_dds_imu_stream.md).

That last confirmation matters most, because the IMU is published by ArduPilot through AP_DDS rather than by the Project AirSim inertial sensor, and no frame conversion is applied anywhere between the ROS message and Basalt. Had ArduPilot published FLU, as much of the ROS ecosystem does, every quaternion in this document would require an additional 180 degree roll.

---

## Step 1 — RPY to Rotation Matrix

Project AirSim uses **intrinsic ZYX Euler angles** (roll, then pitch, then yaw in the parent frame):

    R_B_CA(ψ, θ, φ) = R_z(ψ) · R_y(θ) · R_x(φ)

This is established from source. `JsonUtils::GetTransform` at `core_sim/src/json_utils.cpp:101-128` reads the `xyz` key into the translation and passes `rpy-deg`, converted to radians, to `TransformUtils::ToQuaternion`, whose implementation at `core_sim/src/transforms/transform_utils.cpp:21-51` carries the comment `z-y-x rotation convention (Tait-Bryan angles)` and builds the quaternion in exactly that composition. The key name is `rotation_deg`, resolving to the string `rpy-deg`, and the translation key is the string `xyz`, both at `core_sim/src/constant.hpp:93` and neighbours. The vector order is roll, pitch, yaw, so `"rpy-deg": "0 -45 0"` is φ=0, θ=−45, ψ=0.

One clamp exists on this path and is inert here. `ToQuaternion` at `:25-35` forces any pitch whose magnitude falls between `kEulerSingularityMinor` and `kEulerSingularityMajor`, which are 89.9° and 90.1° per `core_sim/include/core_sim/transforms/transform_utils.hpp:22-24`, onto whichever bound is nearer, to avoid gimbal lock. A camera configured to look straight down at exactly −90° is therefore silently changed to −89.9°, an error of 1.7 mrad. The `front_center` camera at −45° is unaffected, but the clamp must be remembered before any nadir camera is added.

where φ=Roll, θ=Pitch, ψ=Yaw, and:

```
         [1,    0,      0   ]
R_x(φ) = [0,   cosφ,  -sinφ]
         [0,   sinφ,   cosφ ]

         [ cosθ,  0,  sinθ]
R_y(θ) = [  0,   1,    0  ]
         [-sinθ,  0,  cosθ ]

         [cosψ, -sinψ,  0]
R_z(ψ) = [sinψ,  cosψ,  0]
         [  0,    0,    1]
```

---

## Step 2 — Fixed Axis Permutation (simulator cam ↔ OpenCV cam)

The camera NED axes map to OpenCV axes as:

    CA X (forward) → CV Z (optical)
    CA Y (right)   → CV X
    CA Z (down)    → CV Y

Rotation matrices for this fixed permutation:

```
            [0, 1, 0]               [0, 0, 1]
R_CCV_CA =  [0, 0, 1]    R_CA_CCV = [1, 0, 0]  = R_CCV_CA^T
            [1, 0, 0]               [0, 1, 0]
```

---

## Step 3 — Compose T_i_c

Chain of transforms: B from CA from CCV

    T_i_c = T_B_CA · T_CA_CCV

Since T_CA_CCV is a pure rotation (no translation):

    R_i_c = R_B_CA · R_CA_CCV
          = R_z(ψ) · R_y(θ) · R_x(φ)  ·  R_CA_CCV

    t_i_c = [X, Y, Z]ᵀ    (directly from the origin `xyz` key)

**The translation does NOT need inversion.** The `xyz` key is already "camera
origin expressed in the parent link frame", which is exactly what t_i_c represents.

### SE3 Inversion Reference

For the general SE3 inverse (needed if you have the opposite-direction transform):

    If  T = (R, t)  then  T^{-1} = (Rᵀ, −Rᵀ t)

If you are given T_CCV_B = (R_c, t_c), then:

    T_i_c = T_B_CCV = T_CCV_B⁻¹ = (R_cᵀ, −R_cᵀ · t_c)

The term `−R_cᵀ · t_c` is the "inverted translation" — it converts the body-origin
position from camera coordinates back to body coordinates.

---

## Why the parent-link origin is the IMU origin

The chain above yields a camera-to-body extrinsic, and Basalt wants a camera-to-IMU extrinsic. The two coincide because the simulated IMU is unconditionally at the parent link's origin, and this is established by the absence of the mechanism that would allow otherwise.

`Imu::Loader::Load` at `core_sim/src/sensors/imu.cpp:258-268` calls only `LoadImuSettings`, which at `:270-292` reads the `accelerometer` and `gyroscope` objects and nothing else. No call to `JsonUtils::GetTransform` appears anywhere in `imu.cpp`, unlike `camera.cpp:1462-1465`, `distance_sensor.cpp:251`, `magnetometer.cpp:294`, `rotor.cpp:398` and `wheel.cpp:409`, all of which parse an `origin`. An `origin` block placed on an IMU sensor entry would be read by nothing and silently ignored, and `Imu::Impl::Update` at `:185-192` applies no lever-arm or centripetal correction of any kind.

Both the camera and the IMU in `robot_ardu_copter.jsonc` declare `"parent-link": "Frame"`, which is the body origin, so the camera-to-IMU extrinsic is exactly the camera-to-body extrinsic with no residual translation to account for.

---

## Generic Formula Summary

For a camera with parameters (X, Y, Z, Roll=φ, Pitch=θ, Yaw=ψ):

### Full SE3 Composition

The transform chain is evaluated using the SE3 composition rule:

    (R₁, t₁) · (R₂, t₂) = (R₁·R₂,  R₁·t₂ + t₁)

Applied to our two-step chain:

    T_i_c = T_B_CA · T_CA_CCV
          = (R_B_CA, t_xyz) · (R_CA_CCV, 0)
          = (R_B_CA · R_CA_CCV,   R_B_CA · 0 + t_xyz)

The second (axis-permutation) transform has **zero translation** — the camera optical
centre is the same physical point regardless of axis labelling convention. Therefore:

```
R_i_c = R_z(ψ) · R_y(θ) · R_x(φ) · R_CA_CCV

where R_CA_CCV = [[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]]

t_i_c = [X, Y, Z]ᵀ   (NED body frame, metres — taken directly from `xyz`)
```

### When Translation Inversion IS Required

If you are given the **inverse** SE3 — the classical camera extrinsic
`T_CCV_B` that maps body-frame points into camera coordinates:

    T_CCV_B = (R_c, t_c)

then Basalt's T_i_c is the inverse:

    T_i_c = T_B_CCV = T_CCV_B⁻¹ = (R_cᵀ,  −R_cᵀ · t_c)

So:

    R_i_c = R_cᵀ
    t_i_c = −R_cᵀ · t_c        ← "inverted translation"

Here `t_c` is the body-origin expressed in camera coordinates; rotating it by `R_cᵀ`
and negating yields the camera-origin expressed in body coordinates. The `origin` block
provides [X, Y, Z] already in body coordinates, so this inversion step is **not needed**
for the Project AirSim → Basalt case.

### Quaternion from R_i_c (Shepperd's method, assumes trace > −1)

    w = sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    x = (R[2,1] − R[1,2]) / (4w)
    y = (R[0,2] − R[2,0]) / (4w)
    z = (R[1,0] − R[0,1]) / (4w)

Basalt stores quaternions as (qx, qy, qz, qw).

---

## Applied Example: `front_center` Camera

From `/ws/sitl_ws/config/airsim2/robot_ardu_copter.jsonc:313-344`, the sensor entry of type `camera`:

```jsonc
{
  "id": "front_center",
  "type": "camera",
  "parent-link": "Frame",
  "capture-interval": 0.05,
  "origin": {
    "xyz": "0.5 0.0 0.1",
    "rpy-deg": "0 -45 0"
  }
}
```

Parameters: φ=0°, θ=−45°, ψ=0°

### R_y(−45°)

    [[ cos45,  0, -sin45 ],   [[ √2/2,  0, -√2/2 ],
     [   0,    1,    0   ], =  [  0,    1,    0   ],
     [ sin45,  0,  cos45 ]]    [[ √2/2,  0,  √2/2 ]]

    where √2/2 ≈ 0.70711

### R_i_c = R_y(−45°) · R_CA_CCV

    [[ √2/2,  0, -√2/2 ],   [[ 0, 0, 1 ],   [[ 0,    -√2/2,  √2/2 ],
     [  0,    1,    0   ], ·  [  1, 0, 0 ], =  [  1,     0,      0  ],
     [ √2/2,  0,  √2/2 ]]    [[  0, 1, 0 ]]    [[  0,    √2/2,  √2/2 ]]

### Quaternion

    trace = 0 + 0 + √2/2 = 0.70711
    qw = √(1 + 0.70711) / 2 = √1.70711 / 2 ≈ 0.65328
    qx = (R[2,1] − R[1,2]) / (4·0.65328) = ( √2/2 − 0 )     / 2.61313 ≈ 0.27060
    qy = (R[0,2] − R[2,0]) / (4·0.65328) = ( √2/2 − 0 )     / 2.61313 ≈ 0.27060
    qz = (R[1,0] − R[0,1]) / (4·0.65328) = (1 + √2/2)       / 2.61313 ≈ 0.65328

### Resulting Basalt JSON entry

```json
{
    "px":  0.5,
    "py":  0.0,
    "pz":  0.1,
    "qx":  0.27060,
    "qy":  0.27060,
    "qz":  0.65328,
    "qw":  0.65328
}
```

Physical interpretation: the camera is 0.5 m forward and 0.1 m below the IMU,
with its optical axis pointing 45° downward from horizontal. `data/sitl_calib.json`
carries these analytic values.

### Numerical check of the composition

Evaluating R_i_c = R_y(−45°) · R_CA_CCV gives the following.

```
R_i_c = [ 0.00000,  -0.70711,   0.70711 ]
        [ 1.00000,   0.00000,   0.00000 ]
        [ 0.00000,   0.70711,   0.70711 ]

det(R_i_c) = 1.000000
```

The determinant confirms a proper rotation. Reading the columns as the camera axes expressed in the NED body frame gives a direct physical check.

| Camera axis | Direction in NED body frame | Interpretation |
|---|---|---|
| +X, image right | (0, 1, 0) | body right, correct for a zero-roll, zero-yaw mount |
| +Y, image down | (−0.70711, 0, 0.70711) | backward and down, correct for a 45° downward tilt |
| +Z, optical axis | (0.70711, 0, 0.70711) | forward and down at 45°, matching a pitch of −45 |

The quaternion extracted by Shepperd's method is (qx, qy, qz, qw) = (0.27060, 0.27060, 0.65328, 0.65328) with norm 1.00000000, matching the values in `data/sitl_calib.json` to five decimal places.

The pitch sign convention is consistent with this result. Applying R_y(−45°) to the body forward axis yields (0.7071, 0, 0.7071), which points forward and downward because z is down, so a negative pitch tilts the camera toward the ground and a pitch of −90 would point it straight down, subject to the singularity clamp of Step 1.

The translation (0.5, 0.0, 0.1) is taken directly from the `xyz` key and requires no inversion, as argued above. In NED it places the camera 0.5 m forward of and 0.1 m below the body origin, which is the IMU origin by the argument of the preceding section.

---

## Intrinsics

`robot_ardu_copter.jsonc:324-336` sets `width` 640, `height` 480 and `fov-degrees` 90 in the capture settings, an ideal pinhole with no lens distortion.

    fx = (W/2) / tan(FOV_h / 2) = 320 / tan(45°) = 320

With square pixels fy equals fx, and the principal point sits at the image centre, giving cx 320 and cy 240. All four values in `sitl_calib.json:14-30` match, and the `pinhole` camera type is appropriate because the renderer produces an ideal projection.

---

## Frame rate, measured

The configuration requests `capture-interval` of 0.05 s, a nominal 20 Hz. What the topic delivers is materially slower and materially irregular, which bears on both the Basalt frontend and the ORB-SLAM3 configuration that declares a fixed rate.

A capture of 425 consecutive header stamps from `/airsim_node/Copter/front_center_Scene/image` over a short takeoff and landing sequence gives the following.

| Quantity | Value |
|---|---|
| Frames | 425, all stamps distinct |
| Span | 57.147 s |
| Mean rate | 7.42 Hz |
| Minimum interval | 51 ms |
| Median interval | 51 ms |
| Mean interval | 134.8 ms |
| Maximum interval | 1641 ms |

The interval histogram is quantised. The six most common values are 51 ms occurring 221 times, 102 ms occurring 44 times, 153 ms occurring 32 times, 54 ms occurring 20 times, 204 ms occurring 15 times and 255 ms occurring 13 times, and 342 of the 424 intervals are exact multiples of 51 ms.

The 51 ms quantum is the requested 50 ms rounded up to the next scene tick, since `clock.step-ns` in `scene_ardu_copter.jsonc` is 3 ms and 51 ms is seventeen steps, the smallest multiple of 3 not less than 50. The camera therefore cannot run faster than 19.6 Hz in this scene, and every interval that is a multiple of 51 ms is a whole number of missed capture opportunities rather than a jitter. The residual off-quantum values, chiefly 54 ms and 105 ms, arise where a capture straddles a tick boundary.

The achieved 7.42 Hz against a 19.6 Hz ceiling means roughly 62 percent of available frames are dropped, and the tail is long, with twelve intervals exceeding 250 ms and the worst reaching 1.641 s. A gap of that length spans some 274 inertial samples, over which Basalt must preintegrate with no visual correction, and it is exactly the regime in which gyroscope bias error tilts the attitude and leaks horizontal specific force into the position estimate, as derived in [`imu_noise_parameters.md`](imu_noise_parameters.md) Part 7.3.

Two consequences follow for configuration. `ros_ws/src/slam/orb_slam3/config/Monocular/sitl.yaml:28` declares `Camera.fps: 15`, which overstates the achieved rate by a factor of two and should be brought to 7 or 8 from this evidence. And the dropped frames are a throughput problem in the render and bridge path rather than a configuration error, so raising `capture-interval` towards the ceiling would not help and lowering the render load, or accepting the measured rate in the calibration, is the available course.

Note also that these stamps carry raw simulation time, with `sec` values between 1399 and 1456, because the bridge stamps images from the simulator's own `time_stamp` field. The inertial topic does not necessarily share that epoch, and the mismatch is a live hazard treated in [`ap_dds_imu_stream.md`](ap_dds_imu_stream.md).

---

## The one path on which the frame is not NED

Project AirSim's ROS 2 bridge also publishes the simulator's own inertial topic at `.../sensors/IMU1/imu`, and that stream is not NED. `CreateImuPublisher` at `ros/projectairsim_ros2_cpp/src/projectairsim_ros2_cpp_node.cpp:397-416` passes both vectors through `ToRosVector3`, which at `include/projectairsim_ros2_cpp/ros2_conversion_utils.hpp:241-254` negates the second and third components. That is a 180° rotation about x, taking NED to forward-left-up, and not the yaw-and-flip that would produce a REP 103 east-north-up frame. Anything consuming that topic sees x forward, y left, z up.

Basalt is pointed at the ArduPilot topic and not at this one, so the extrinsic above is the operative one. Should Basalt ever be repointed, `T_imu_cam` must be re-expressed by left-multiplying the rotation by `R_x(180°) = diag(1, −1, −1)` and rotating the translation by the same matrix.

```
R'_i_c = R_x(180°) · R_i_c = [  0.00000,  -0.70711,   0.70711 ]
                             [ -1.00000,   0.00000,   0.00000 ]
                             [  0.00000,  -0.70711,  -0.70711 ]

det = 1.000000
t'_i_c = (0.5, 0.0, -0.1)
```

The corresponding Basalt entry is the following, with the quaternion norm verified as 1.00000000.

```json
{
    "px":  0.5,   "py":  0.0,   "pz": -0.1,
    "qx": -0.65328, "qy": 0.65328, "qz": -0.27060, "qw": 0.27060
}
```

The physical check is that the optical axis, the third column, becomes `(0.70711, 0, −0.70711)`, which in a z-up frame is forward and 45° down, the same physical direction as before. Making the move is deliberately not undertaken, for the reasons recorded in [`/ws/context/sitl-time-synchronisation.md`](../../../../../context/sitl-time-synchronisation.md) under "A deliberate non-goal", and the figures are recorded here so that it can be made without re-deriving them.

---

## Caveats

1. The NED premise is verified at both ends, at the simulator by the forward and upward unit vectors and at the subscriber by the frame identifier and the stationary specific force. If the UE5 or ArduPilot frame convention changes, R_CA_CCV and every derived quaternion must be revisited.

2. Single-camera case. `sitl_calib.json` declares one camera. The formula extends to stereo by applying the derivation independently per camera.

3. Stereo time offset. If two cameras are used, `cam_time_offset_ns` must be determined separately by hardware synchronisation or software alignment. It is currently zero, which is correct for the single-camera configuration, and in any case the line that would apply it at run time is commented out at `src/vi_estimator/sqrt_keypoint_vio.cpp:231`.

4. The extrinsics are not implicated in the stationary trajectory divergence investigated on this stack. That investigation identified the inertial noise parameters, the image ingest path and the timestamp epochs as the load-bearing defects, recorded in [`ap_dds_imu_stream.md`](ap_dds_imu_stream.md) and [`imu_noise_parameters.md`](imu_noise_parameters.md).

5. A stale set of figures exists that must be recognised rather than used. `sitl_calib.json` previously held a T_imu_cam of `px≈-0.017, py≈-0.069, pz≈0.005` with `qz≈0.702, qw≈0.712`, obtained from an earlier empirical calibration run against a different camera mounting, namely a near-90° yaw rotation with a small translation, which does not correspond to the `front_center` camera described here. They are recorded so that a reader who meets them in git history, or in a calibration file copied from an older branch, identifies them as stale rather than mistaking them for an alternative valid calibration.

---

## Reference: Basalt T_i_c Serialization Format

In Basalt calibration JSON, `T_imu_cam` is stored per camera as a quaternion + translation:

```json
"T_imu_cam": [
  {
    "px": <t_x>,   "py": <t_y>,   "pz": <t_z>,
    "qx": <q_x>,   "qy": <q_y>,   "qz": <q_z>,   "qw": <q_w>
  }
]
```

The quaternion is in Hamilton convention (qw = scalar part).

## References

- Engel, Usenko, Cremers, "A Photometrically Calibrated Benchmark For Monocular Visual Odometry", arXiv:1607.02555 (2016) — photometric model context
- Usenko et al., "Visual-Inertial Mapping with Non-Linear Factor Recovery", RA-L 2020, arXiv:1904.06504 — Basalt main paper
- Project AirSim `docs/config_robot.md` — sensor `origin` semantics and the rotor
  `normal-vector` convention at line 615; note that the linked "IMU Settings" page at
  line 712 does not exist in the tree, so `core_sim/src/sensors/imu.cpp` is the only
  authority for the inertial sensor
- Project AirSim transform implementation: `core_sim/src/json_utils.cpp` and
  `core_sim/src/transforms/transform_utils.cpp` — the authoritative source for the
  z-y-x Euler convention and the pitch singularity clamp
- Shepperd, S.W. (1978), "Quaternion from rotation matrix", AIAA J. Guidance and Control
