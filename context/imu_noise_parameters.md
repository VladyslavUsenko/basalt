# IMU Noise Parameters: CosysAirSim ↔ Basalt

## Overview

This document covers the theoretical background for each IMU noise parameter in
CosysAirSim's `settings.json` and their equivalents in Basalt's calibration JSON.
It derives unit conversions and numerical values for the SITL setup.

---

## The Standard IMU Measurement Model

A MEMS IMU produces measurements corrupted by two distinct noise processes:

    ω_meas(t) = ω_true(t)  +  b_g(t)  +  η_g(t)
    a_meas(t) = a_true(t)  +  b_a(t)  +  η_a(t)

where:
- `η_g`, `η_a` — additive **white noise** (broadband, uncorrelated)
- `b_g`, `b_a` — slowly time-varying **biases** with their own stochastic dynamics

These are characterised by separate parameters because they have different statistical
structure and affect state estimation differently.

---

## Part 1 — White Noise Parameters

### 1.1 Angular Random Walk (ARW) — Gyroscope White Noise

**CosysAirSim field**: `AngularRandomWalk`
**Typical unit in AirSim**: °/√hr

#### Physical Meaning

ARW describes the **white noise** on the angular rate measurement. Integrating white
noise produces a random walk in attitude. Over an averaging interval τ, the 1-σ angle
uncertainty grows as:

    σ_θ(τ) = ARW × √τ   [degrees when τ in hours]

This is the -½ slope region in an Allan Deviation (ADEV) plot. It is also called the
**Angle Random Walk** for this reason.

#### Mathematical Model

The gyro measurement noise η_g(t) is zero-mean white Gaussian with Power Spectral
Density (PSD):

    S_η_g(f) = σ_c_g²   [(rad/s)² / Hz  =  rad²/s]

where σ_c_g is the **continuous-time gyroscope noise standard deviation** (the Basalt
parameter). This is related to ARW by:

    σ_θ²(T) = σ_c_g² × T   [rad² for T in seconds]

Setting T = 1 hr = 3600 s and converting ARW from °/√hr to rad/√hr:

    σ_c_g² × 3600 = (ARW × π/180)²

    σ_c_g = ARW × π / (180 × 60)   [rad/s / √Hz]

#### Unit Conversion

    σ_c_g [rad/s/√Hz]  =  ARW [°/√hr]  ×  π / (180 × 60)
                        =  ARW [°/√hr]  ×  2.909 × 10⁻⁴

#### Basalt Equivalent

    Basalt field:   gyro_noise_std   [rad/s/√Hz = rad/√s]

    gyro_noise_std = ARW × π / (180 × 60)

#### SITL Numerical Value

    ARW = 0.3 °/√hr
    gyro_noise_std = 0.3 × 2.909e-4 ≈ 8.73e-5  rad/s/√Hz

#### How Basalt Uses It

`calibration.hpp` converts to a discrete-time standard deviation for pre-integration:

    σ_discrete_g = gyro_noise_std × √(imu_update_rate)   [rad/s]

This is the noise standard deviation per IMU sample used to build the pre-integration
covariance (see `preintegration.h:177-179`):

    Cov += G · diag(gyro_cov) · Gᵀ

where `gyro_cov = dicrete_time_gyro_noise_std².array()`.

---

### 1.2 Velocity Random Walk (VRW) — Accelerometer White Noise

**CosysAirSim field**: `VelocityRandomWalk`
**Typical unit in AirSim**: m/s/√hr

#### Physical Meaning

VRW is the accelerometer equivalent of ARW. Integrating accelerometer white noise
produces a random walk in velocity:

    σ_v(τ) = VRW × √τ   [m/s when τ in hours]

This is also called the **Velocity Random Walk** for this reason.

#### Mathematical Model

The accelerometer noise η_a(t) has PSD:

    S_η_a(f) = σ_c_a²   [(m/s²)² / Hz  =  m²/s³]

By the same derivation as ARW:

    σ_v²(T) = σ_c_a² × T

Setting T = 1 hr:

    σ_c_a² × 3600 = VRW²
    σ_c_a = VRW / 60   [m/s² / √Hz]

#### Unit Conversion

    σ_c_a [m/s²/√Hz]  =  VRW [m/s/√hr]  ÷  60

#### Basalt Equivalent

    Basalt field:   accel_noise_std   [m/s²/√Hz = m/s/√s]

    accel_noise_std = VRW / 60

#### SITL Numerical Value

    VRW = 0.24 m/s/√hr
    accel_noise_std = 0.24 / 60 = 4.0e-3  m/s²/√Hz

#### How Basalt Uses It

Same pattern as gyro:

    σ_discrete_a = accel_noise_std × √(imu_update_rate)   [m/s²]

---

## Part 2 — Bias Parameters

### 2.1 The Two Bias Models Compared

#### CosysAirSim: First-Order Gauss-Markov (FOGM) Process

AirSim models bias as an Ornstein–Uhlenbeck process (mean-reverting random walk):

    ḃ(t) = -b(t)/τ  +  σ_w × w(t)

where:
- τ = `BiasStabilityTau` (seconds) — **time constant** of mean-reversion
- σ_w = driving noise intensity [units/√s]
- w(t) = unit white noise
- Steady-state variance: E[b²] = σ_w² × τ/2  →  σ_b = σ_w × √(τ/2)

The parameter `BiasStability` (σ_b) is the **steady-state standard deviation** of
the bias — i.e., how large the bias gets on average over a very long time.

#### Basalt: Continuous-Time Random Walk

Basalt uses the simpler **Rate Random Walk** (RRW) model, also known as Wiener process:

    ḃ(t) ~ N(0, σ_rw²)   per unit time

Over interval Δt, bias evolves as: b(t+Δt) - b(t) ~ N(0, σ_rw² × Δt)

This is a limiting case of FOGM when Δt << τ (the bias drifts freely between FOGM
mean-reversion events). In this regime, the two models are equivalent with:

    σ_rw = σ_b × √(2/τ)   [units/√s]

This is the **conversion formula** between AirSim FOGM and Basalt RRW parameters.

#### Relationship to Allan Deviation

In an Allan Deviation (ADEV) plot:
- **Bias Instability (BI)**: appears as the flat minimum region; corresponds to 1/f
  flicker noise. `BiasStability` ≈ BI value at the ADEV minimum.
- **Rate Random Walk (RRW)**: appears as the +½ slope region at longer averaging
  times; this is what Basalt `bias_std` models.
- **FOGM** approximates the transition between BI and RRW; τ marks the boundary.

The conversion `σ_rw = σ_b × √(2/τ)` comes from matching the FOGM PSD to a
two-sided approximation of the 1/f + RRW spectrum.

---

### 2.2 Gyroscope Bias — GyroBiasStability + GyroBiasStabilityTau

**CosysAirSim fields**:
- `GyroBiasStability` — steady-state bias std, unit: **°/hr**
- `GyroBiasStabilityTau` — FOGM time constant, unit: **seconds**

#### Unit Conversion

Step 1 — Convert GyroBiasStability to SI:

    σ_b_g [rad/s]  =  GyroBiasStability [°/hr]  ×  π / (180 × 3600)
                    =  GyroBiasStability [°/hr]  ×  4.848 × 10⁻⁶

Step 2 — Convert FOGM to RRW (Basalt model):

    gyro_bias_std [rad/s/√s]  =  σ_b_g × √(2 / GyroBiasStabilityTau)

#### Basalt Equivalent

    Basalt field:   gyro_bias_std   [rad/s/√s = rad/s²/√Hz]

#### SITL Numerical Value

    GyroBiasStability = 4.6 °/hr
    GyroBiasStabilityTau = 500 s

    σ_b_g = 4.6 × 4.848e-6 = 2.229e-5 rad/s
    gyro_bias_std = 2.229e-5 × √(2/500) = 2.229e-5 × 6.325e-2 ≈ 1.41e-6  rad/s/√s

#### How Basalt Uses It

In `sqrt_keypoint_vio.cpp:116`:

    gyro_bias_sqrt_weight = calib.gyro_bias_std.array().inverse()

This is used as the information weight (1/σ) on the bias prior/regularization in the
factor graph. A smaller `gyro_bias_std` means Basalt trusts the bias to stay fixed
— tighter regularization.

---

### 2.3 Accelerometer Bias — AccelBiasStability + AccelBiasStabilityTau

**CosysAirSim fields**:
- `AccelBiasStability` — steady-state bias std, unit: **μg** (micro-g, where g = 9.81 m/s²)
- `AccelBiasStabilityTau` — FOGM time constant, unit: **seconds**

#### Unit Conversion

Step 1 — Convert AccelBiasStability to SI:

    σ_b_a [m/s²]  =  AccelBiasStability [μg]  ×  9.81 × 10⁻⁶

Step 2 — Convert FOGM to RRW:

    accel_bias_std [m/s²/√s]  =  σ_b_a × √(2 / AccelBiasStabilityTau)

#### Basalt Equivalent

    Basalt field:   accel_bias_std   [m/s²/√s = m/s³/√Hz... i.e. m/s²/√Hz when in √s]

    Note: the unit is sometimes written as m/s²/√s to emphasize it is a
    continuous-time rate-of-change noise; per unit root-second of elapsed time,
    the bias drifts by accel_bias_std m/s².

#### SITL Numerical Value

    AccelBiasStability = 36 μg
    AccelBiasStabilityTau = 800 s

    σ_b_a = 36 × 9.81e-6 = 3.532e-4 m/s²
    accel_bias_std = 3.532e-4 × √(2/800) = 3.532e-4 × 5.0e-2 ≈ 1.77e-5  m/s²/√s

#### How Basalt Uses It

Same as gyro_bias_std — used as the inverse weight on the bias prior:

    accel_bias_sqrt_weight = calib.accel_bias_std.array().inverse()

---

## Part 3 — IMU Update Rate

**CosysAirSim**: The IMU publishes at the simulation tick rate (or overridden via
sensors config). For ArduCopter SITL, the IMU runs at 1000 Hz by default (controlled
by ArduPilot, not AirSim).

**Basalt field**: `imu_update_rate` [Hz]

This scalar is used only in the discrete-time conversion:

    σ_discrete = σ_continuous × √(imu_update_rate)

It does NOT affect the underlying physical noise parameters — only how Basalt maps
the continuous-time PSD to per-sample noise. If the IMU rate changes, only this
field needs updating; the noise std fields remain the same.

---

## Part 4 — GenerateNoise Flag

**CosysAirSim field**: `GenerateNoise: false`

**CRITICAL**: With `GenerateNoise: false` (current SITL config), ALL noise
parameters above are **ignored**. The simulated IMU outputs perfectly clean
measurements with no noise and no bias drift.

To activate simulation noise:
```json
"Imu": {
    "GenerateNoise": true,
    ...noise parameters...
}
```

Implications for SLAM:
- With noise disabled, VIO will work even with coarse Basalt IMU parameters.
- Real-world deployment requires re-calibrating all four noise parameters.
- For testing IMU integration correctness, enable noise and verify SLAM degrades
  gracefully as noise increases.

---

## Part 5 — Complete Conversion Table

### SITL Config → Basalt Calibration Values

| CosysAirSim Field | Value | AirSim Unit | Basalt Field | Basalt Value | Basalt Unit | Conversion |
|---|---|---|---|---|---|---|
| `AngularRandomWalk` | 0.3 | °/√hr | `gyro_noise_std` | **8.73e-5** | rad/s/√Hz | × π/(180×60) |
| `VelocityRandomWalk` | 0.24 | m/s/√hr | `accel_noise_std` | **4.00e-3** | m/s²/√Hz | ÷ 60 |
| `GyroBiasStability` + `GyroBiasStabilityTau` | 4.6 + 500 | °/hr + s | `gyro_bias_std` | **1.41e-6** | rad/s/√s | σ_b[rad/s] × √(2/τ) |
| `AccelBiasStability` + `AccelBiasStabilityTau` | 36 + 800 | μg + s | `accel_bias_std` | **1.77e-5** | m/s²/√s | σ_b[m/s²] × √(2/τ) |

### Resulting Basalt Calibration JSON Block (SITL)

```json
"imu_update_rate": 1000.0,
"gyro_noise_std":  [8.73e-5,  8.73e-5,  8.73e-5],
"accel_noise_std": [4.00e-3,  4.00e-3,  4.00e-3],
"gyro_bias_std":   [1.41e-6,  1.41e-6,  1.41e-6],
"accel_bias_std":  [1.77e-5,  1.77e-5,  1.77e-5]
```

### Comparison: EuRoC vs SITL

| Parameter | EuRoC Default (ADIS16448) | SITL (CosysAirSim) | Ratio |
|---|---|---|---|
| `gyro_noise_std` (rad/s/√Hz) | 2.82e-4 | 8.73e-5 | ~3× better simulation |
| `accel_noise_std` (m/s²/√Hz) | 1.60e-2 | 4.00e-3 | ~4× better simulation |
| `gyro_bias_std` (rad/s/√s) | 1.00e-4 | 1.41e-6 | ~70× more stable |
| `accel_bias_std` (m/s²/√s) | 1.00e-3 | 1.77e-5 | ~56× more stable |

The simulation models a considerably better IMU than the EuRoC sensor. This means
the VIO will rely more on vision and less on IMU prediction between frames — the IMU
term in the factor graph will have lower weight.

---

## Part 6 — Allan Deviation Plot Reference

A real IMU ADEV log-log plot shows distinct regions corresponding to each parameter:

```
log σ_ADEV
│
│   ╲ ARW/VRW region          ╱  RRW region
│    ╲ slope = −½     __BI__/   slope = +½
│     ╲               (flat)  /
│      ╲____________________/
│
└─────────────────────────────── log τ (averaging time)
   τ_short                τ_long
                    ↑
             BiasStabilityTau (where FOGM transitions to RRW)
```

| ADEV Region | Slope | AirSim Parameter | Basalt Parameter |
|---|---|---|---|
| Short τ, slope −½ | -0.5 | `AngularRandomWalk` / `VelocityRandomWalk` | `gyro_noise_std` / `accel_noise_std` |
| Middle τ, flat | 0 | `GyroBiasStability` / `AccelBiasStability` | (approximated via conversion) |
| Long τ, slope +½ | +0.5 | Both + Tau (FOGM tail) | `gyro_bias_std` / `accel_bias_std` |

---

## References

- IEEE Std 952-1997, "IEEE Standard Specification Format Guide and Test Procedure for
  Single-Axis Interferometric Fiber Optic Gyros" — defines ARW, BI, RRW in Allan Variance.
- Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation", TR 2005-002
  — continuous vs discrete IMU noise models.
- Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry",
  IEEE T-RO 2017 — IMU preintegration covariance with continuous-time noise model.
- Usenko et al., "Visual-Inertial Mapping with Non-Linear Factor Recovery",
  RA-L 2020 (arXiv:1904.06504) — Basalt's factor structure and IMU weighting.
- CosysAirSim IMU implementation: `AirLib/include/sensors/imu/ImuSimpleParams.hpp`
