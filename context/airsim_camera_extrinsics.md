# AirSim Camera Extrinsics → Basalt T_imu_cam

## Problem Statement

AirSim's `settings.json` defines each camera by:
- **Position** `[X, Y, Z]` — camera origin in the NED body frame (metres)
- **Orientation** `[Pitch, Roll, Yaw]` — camera frame orientation relative to body frame (degrees)

Basalt stores `T_imu_cam` (= `T_i_c`) per camera: the SE3 that maps a point from the
**OpenCV camera frame** into the **IMU / body frame**:

    p_body = R_i_c · p_cam + t_i_c

This document derives the formula and applies it to the `front_center` camera.

---

## Coordinate Frames

| Symbol | Name | Convention |
|--------|------|------------|
| **B** | Body / IMU | NED: X forward, Y right, Z down |
| **CA** | AirSim camera | Same as B when Pitch=Roll=Yaw=0 (NED) |
| **CCV** | Basalt / OpenCV camera | X right, Y down, Z forward (optical axis) |

---

## Step 1 — RPY to Rotation Matrix

AirSim uses **intrinsic ZYX Euler angles** (roll, then pitch, then yaw in body frame):

    R_B_CA(ψ, θ, φ) = R_z(ψ) · R_y(θ) · R_x(φ)

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

## Step 2 — Fixed Axis Permutation (AirSim cam ↔ OpenCV cam)

AirSim camera NED axes map to OpenCV axes as:

    AirSim X (forward) → CV Z (optical)
    AirSim Y (right)   → CV X
    AirSim Z (down)    → CV Y

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

    t_i_c = [X, Y, Z]ᵀ    (directly from AirSim settings.json)

**The translation does NOT need inversion.** AirSim [X, Y, Z] is already "camera
origin expressed in body frame", which is exactly what t_i_c represents.

### SE3 Inversion Reference

For the general SE3 inverse (needed if you have the opposite-direction transform):

    If  T = (R, t)  then  T^{-1} = (Rᵀ, −Rᵀ t)

If you are given T_CCV_B = (R_c, t_c), then:

    T_i_c = T_B_CCV = T_CCV_B⁻¹ = (R_cᵀ, −R_cᵀ · t_c)

The term `−R_cᵀ · t_c` is the "inverted translation" — it converts the body-origin
position from camera coordinates back to body coordinates.

---

## Generic Formula Summary

For a camera with AirSim parameters (X, Y, Z, Roll=φ, Pitch=θ, Yaw=ψ):

### Full SE3 Composition

The transform chain is evaluated using the SE3 composition rule:

    (R₁, t₁) · (R₂, t₂) = (R₁·R₂,  R₁·t₂ + t₁)

Applied to our two-step chain:

    T_i_c = T_B_CA · T_CA_CCV
          = (R_B_CA, t_AirSim) · (R_CA_CCV, 0)
          = (R_B_CA · R_CA_CCV,   R_B_CA · 0 + t_AirSim)

The second (axis-permutation) transform has **zero translation** — the camera optical
centre is the same physical point regardless of axis labelling convention. Therefore:

```
R_i_c = R_z(ψ) · R_y(θ) · R_x(φ) · R_CA_CCV

where R_CA_CCV = [[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]]

t_i_c = [X, Y, Z]ᵀ   (NED body frame, metres — taken directly from AirSim)
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
and negating yields the camera-origin expressed in body coordinates. AirSim provides
[X, Y, Z] already in body coordinates, so this inversion step is **not needed** for
the AirSim → Basalt case.

### Quaternion from R_i_c (Shepperd's method, assumes trace > −1)

    w = sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    x = (R[2,1] − R[1,2]) / (4w)
    y = (R[0,2] − R[2,0]) / (4w)
    z = (R[1,0] − R[0,1]) / (4w)

Basalt stores quaternions as (qx, qy, qz, qw).

---

## Applied Example: `front_center` Camera

From `/ws/envs/ue5/AirSim/Unreal/Environments/settings.json`:

```json
"front_center": {
    "X": 0.50, "Y": 0, "Z": 0.10,
    "Pitch": -45, "Roll": 0, "Yaw": 0
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
with its optical axis pointing 45° downward from horizontal.

---

## Caveats and Validation

1. **AirSim NED assumption**: verified — AirSim reports body-frame positions in NED.
   If UE5 coordinate system changes (e.g., ENU body frame), R_CA_CCV must be updated.

2. **Single-camera case**: `sitl_calib.json` has one camera. The formula extends
   trivially to stereo by applying the same derivation independently per camera.

3. **Experimental calibration vs. analytic**: the `sitl_calib.json` T_imu_cam
   (`px≈-0.017, py≈-0.069, pz≈0.005`, `qz≈0.702, qw≈0.712`) was obtained from
   an earlier empirical calibration run and reflects a different camera mounting
   (near-90° yaw rotation, small translation). It does NOT correspond to the
   front_center camera above. Use the formula in this document for the current setup.

4. **Stereo time offset**: if two cameras are used, `cam_time_offset_ns` must be
   separately determined (hardware sync or software alignment).

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
- AirSim docs: https://microsoft.github.io/AirSim/settings/ — camera pose convention
- Shepperd, S.W. (1978), "Quaternion from rotation matrix", AIAA J. Guidance and Control
