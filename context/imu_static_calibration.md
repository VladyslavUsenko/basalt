# IMU Static Calibration: calib_accel_bias & calib_gyro_bias

## 1. What Are These Parameters?

`calib_accel_bias` (9 params) and `calib_gyro_bias` (12 params) in Basalt's calibration
JSON are **static hardware calibration** parameters. They encode deterministic,
repeatable sensor errors due to manufacturing imperfections:

| Error Type | What causes it | Parameter(s) |
|---|---|---|
| Additive offset bias | Sensor output ≠ 0 at zero input | params 0–2 (b_x, b_y, b_z) |
| Scale factor error | Output gain ≠ 1.0 | diagonal of params 3–8/11 |
| Axis misalignment | Sensor axes not orthogonal | off-diagonal of params 3–8/11 |

These are **not** the VIO's runtime dynamic bias (b_g, b_a) — that is separately
estimated by the factor graph during operation. The static calib is applied first
as a preprocessing step before any data enters the VIO.

---

## 2. Mathematical Model

### 2.1 Accelerometer — 9-Parameter Model (`CalibAccelBias`)

Source: `thirdparty/basalt-headers/include/basalt/calibration/calib_bias.hpp`

```
a_c = (I + S_a) · a_r  −  b_a
```

where:
- `a_r` ∈ ℝ³ — raw accelerometer measurement (m/s²)
- `a_c` ∈ ℝ³ — calibrated measurement passed to the VIO
- `b_a` = [b_x, b_y, b_z]ᵀ — additive bias vector (params 0–2)
- `S_a` — lower-triangular scale/misalignment matrix (params 3–8)

The parameter vector and its layout in S_a:

    params = [b_x, b_y, b_z,  s1, s2, s3, s4, s5, s6]
                ↑ bias (3)        ↑ scale/misalignment (6)

    S_a = [[ s1,  0,   0  ],    (I + S_a) diagonal ≈ [1+s1, 1+s4, 1+s6]
            [ s2,  s4,  0  ],
            [ s3,  s5,  s6 ]]

Lower-triangular structure: the x-axis measurement is assumed to be the
reference axis. Cross-coupling terms only appear where later axes are affected
by earlier axis measurements.

Implementation in `getBiasAndScale()` (`calib_bias.hpp:91-98`):
```cpp
accel_bias = accel_bias_full_.head<3>();          // params 0-2 → b_a
accel_scale.col(0) = accel_bias_full_.segment<3>(3);  // params 3,4,5 → col 0 of S
accel_scale(1, 1) = accel_bias_full_(6);              // param 6 → S[1,1]
accel_scale(2, 1) = accel_bias_full_(7);              // param 7 → S[2,1]
accel_scale(2, 2) = accel_bias_full_(8);              // param 8 → S[2,2]
```

Calibrated measurement (`calib_bias.hpp:106-112`):
```cpp
return (raw_measurement + accel_scale * raw_measurement - accel_bias);
// = (I + S_a) * a_r - b_a
```

When all params are zero: `a_c = a_r` (identity — no correction applied).

---

### 2.2 Gyroscope — 12-Parameter Model (`CalibGyroBias`)

```
ω_c = (I + S_g) · ω_r  −  b_g
```

where:
- `ω_r` ∈ ℝ³ — raw gyroscope measurement (rad/s)
- `ω_c` ∈ ℝ³ — calibrated measurement passed to VIO
- `b_g` = [b_x, b_y, b_z]ᵀ — additive bias vector (params 0–2)
- `S_g` — **full** 3×3 scale/misalignment matrix (params 3–11, all 9 entries)

    params = [b_x, b_y, b_z,  s1, s2, s3,  s4, s5, s6,  s7, s8, s9]
                ↑ bias (3)        ↑ col 0        ↑ col 1        ↑ col 2

    S_g = [[ s1, s4, s7 ],
            [ s2, s5, s8 ],
            [ s3, s6, s9 ]]

The gyroscope uses a **full** (not lower-triangular) matrix because gyroscopes
are subject to g-sensitivity (accelerometer cross-coupling) in addition to
axis misalignment, making all 9 off-diagonal terms potentially non-zero.

Implementation (`getBiasAndScale()`, `calib_bias.hpp:179-183`):
```cpp
gyro_bias = gyro_bias_full_.head<3>();
gyro_scale.col(0) = gyro_bias_full_.segment<3>(3);   // params 3-5
gyro_scale.col(1) = gyro_bias_full_.segment<3>(6);   // params 6-8
gyro_scale.col(2) = gyro_bias_full_.segment<3>(9);   // params 9-11
```

---

## 3. Where and When These Are Applied in Basalt

### 3.1 Applied to every raw IMU sample — static preprocessing

In `sqrt_keypoint_vio.cpp:226-230`, and repeated after every IMU pop in `ProcessFrame`. Corrected 2026-08-23. This previously cited `sqrt_keypoint_vio.cpp:170-173`, the copy inside the producer consumer `proc_func` lambda, which was removed as redundant on that date because `ProcessFrame` acquires and calibrates the first sample itself under the identical guard.
```cpp
imuData->accel = this->calib.calib_accel_bias.getCalibrated(imuData->accel);
imuData->gyro  = this->calib.calib_gyro_bias.getCalibrated(imuData->gyro);
```

This happens **before** pre-integration and **before** the VIO factor graph sees
any data. The VIO never sees raw measurements — only calibrated ones.

### 3.2 Bias vector seeds the initial VIO dynamic bias

In `basalt_slam.cpp:207-214`:
```cpp
calib.calib_gyro_bias.getBiasAndScale(bg, sg);   // extracts first 3 params
calib.calib_accel_bias.getBiasAndScale(ba, sa);

controller->initialize(0, Sophus::SE3d(), Eigen::Vector3d::Zero(),
                        bg,   // ← initial gyro bias for factor graph
                        ba);  // ← initial accel bias for factor graph
```

The first 3 parameters of each calibration vector (`b_g`, `b_a`) seed the VIO's
dynamic bias state. The VIO then refines this online. A wrong initial bias slows
convergence; a very wrong initial bias can cause divergence in the first few frames.

### 3.3 Optimized during IMU-camera calibration

In `spline_linearize.h:217-219` and `spline_optimize.h:550-554`, these parameters
appear in the Jacobians of the IMU residuals:

```
gyro residual:  r_g = ω_spline(t) − ω_c(t) = ω_spline(t) − (I+S_g)ω_r(t) + b_g
accel residual: r_a = R⁻¹(a_world(t) + g) − a_c(t)
```

During `basalt_calibrate_imu`, they are jointly optimized with T_i_c, camera
intrinsics, and the B-spline trajectory.

---

## 4. Relationship to Dynamic VIO Bias

| | Static calibration (`calib_bias`) | Dynamic VIO bias (bg, ba) |
|---|---|---|
| **What it models** | Deterministic hardware offsets (manufacturing) | Slowly time-varying residual drift |
| **How determined** | Off-line calibration (static recording + optimisation) | Estimated online by factor graph |
| **When applied** | Pre-processing, every IMU sample | Inside pre-integration covariance |
| **Update rate** | Once (at calibration time) | Every keyframe in the BA window |
| **Parameter count** | 9 (accel) + 12 (gyro) | 3+3 (bias 3-vectors only) |
| **Location in code** | `calib.calib_accel_bias`, `.calib_gyro_bias` | VIO state `bias_accel`, `bias_gyro` |

The dynamic bias starts from the static calib value (`bg_init = b_g` from calib),
then drifts under the `gyro_bias_std` random walk model.

---

## 5. EuRoC Dataset Values

EuRoC (`data/euroc_ds_calib.json`) was calibrated with the ADIS16448 IMU.
Typical values after calibration:

```json
"calib_accel_bias": [
    -0.003025405479279035,   ← b_x ≈ −3.0 mm/s² offset
     0.1200005286487319,     ← b_y ≈ 120 mm/s² offset (significant!)
     0.06708820471592454,    ← b_z ≈ 67 mm/s² offset
     0.0, 0.0, 0.0,          ← scale col 0 (not calibrated → zeros)
     0.0, 0.0, 0.0           ← scale remaining (not calibrated → zeros)
],
"calib_gyro_bias": [
    -0.002186848441668376,   ← b_x ≈ −2.2 mrad/s
     0.020427823167917037,   ← b_y ≈  20.4 mrad/s
     0.07668367023977922,    ← b_z ≈  76.7 mrad/s (large!)
     0.0, ..., 0.0           ← 9 scale/misalignment terms (zeros)
]
```

Observations:
- All scale/misalignment parameters are zero — the EuRoC calibration only estimated
  the bias offset vector, not full scale calibration.
- The accel b_y = 0.12 m/s² is a large deterministic offset (≈ 1.2% of g).
- The gyro b_z = 77 mrad/s ≈ 4.4°/s — would cause obvious heading drift if uncorrected.

---

## 6. SITL Simulator Values

For a simulated IMU (Project AirSim), all parameters must be **zero**, because `turn-on-bias` is configured to zero and the sensor model contains no scale-factor or misalignment term at all:

```json
"calib_accel_bias": [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
"calib_gyro_bias":  [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
```

Reasons:
1. The simulator generates measurements directly from ground truth — there are no
   manufacturing offsets to correct.
2. Bias drift is simulated stochastically as a pure Wiener random walk, always on since
   Project AirSim has no flag that disables the noise model, but this is a dynamic bias
   handled by the VIO, not a static offset. See [`imu_noise_parameters.md`](imu_noise_parameters.md)
   section 2.1b.
3. Non-zero static calibration would **over-correct** the simulator output, introducing
   artificial systematic error.

**Bug in earlier `sitl_calib.json`**: values were copied from EuRoC's ADIS16448
calibration and were non-zero. This was incorrect and has been fixed (zeroed).

---

## 7. How to Calibrate a Real IMU

### Step 1 — Collect a static IMU recording

Mount the IMU rigidly. Record several minutes of data at rest in multiple orientations
(to separate bias from gravity projection). For Allan Variance, 2+ hours is ideal.

### Step 2 — Run `basalt_calibrate_imu`

```bash
basalt_calibrate_imu \
    --dataset-path /path/to/static_recording \
    --dataset-type euroc \
    --aprilgrid /path/to/aprilgrid.json \
    --result-path /path/to/output \
    --gyro-noise-std  0.0005818 \   # from imu_noise_parameters.md
    --accel-noise-std 0.020     \
    --gyro-bias-std   7.92e-6   \
    --accel-bias-std  2.83e-4
```

The tool (`src/calibrate_imu.cpp` → `CamImuCalib`) fits a B-spline through the
IMU trajectory and jointly optimizes:
- `calib_accel_bias` (9 params)
- `calib_gyro_bias` (12 params)
- T_i_c (camera-IMU extrinsics)
- Camera intrinsics
- Gravity vector direction

The result is saved as a calibration JSON which provides values for all fields
in `sitl_calib.json` / your custom calib file.

### Step 3 — Verify

After running, verify:
- Bias offsets (params 0–2) are << 1 m/s² for accel, << 0.1 rad/s for gyro.
- Scale/misalignment terms (params 3+) are << 0.05 (5% deviation from identity).
- Large values suggest noisy calibration data or a defective sensor.

---

## 8. JSON Serialization Format

Serialized by cereal in `headers_serialization.h:280-281`:
```cpp
cereal::make_nvp("calib_accel_bias", cam.calib_accel_bias.getParam()),
cereal::make_nvp("calib_gyro_bias",  cam.calib_gyro_bias.getParam())
```

The flat arrays map to the parameter vectors as described in Section 2:
- `calib_accel_bias`: 9-element array `[b_x, b_y, b_z, s1, s2, s3, s4, s5, s6]`
- `calib_gyro_bias`: 12-element array `[b_x, b_y, b_z, s1..s3, s4..s6, s7..s9]`

---

## 9. Summary: Do I Need to Set These?

| Scenario | calib_accel_bias | calib_gyro_bias |
|---|---|---|
| **SITL (Project AirSim, ideal IMU)** | All zeros ✓ | All zeros ✓ |
| **SITL with stochastic noise live** | All zeros (dynamic bias handles drift) | All zeros |
| **Real IMU, first bring-up** | All zeros (VIO will converge from cold start) | All zeros |
| **Real IMU, after calibration** | Output from `basalt_calibrate_imu` | Output from same |
| **Copying from another sensor** | **Do NOT** — sensor-specific, not portable | **Do NOT** |

For the Basalt SITL setup: **leave all 21 parameters at zero**.
