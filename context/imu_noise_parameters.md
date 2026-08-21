# IMU Noise Parameters, Project AirSim to Basalt

## Overview

This document covers the theoretical background for each IMU noise parameter in the Project AirSim robot configuration and their equivalents in Basalt's calibration JSON. It establishes the mapping between the two, derives the numerical values for the SITL setup, and records the three effects that ArduPilot interposes between the simulated sensor and the samples Basalt actually receives.

The simulator of record is Project AirSim, configured by `/ws/sitl_ws/config/airsim2/robot_ardu_copter.jsonc`, which `djinn start sitl2` runs. The single most consequential fact about that configuration is that Project AirSim reads all four noise parameters in SI units directly and performs no unit conversion whatever at load time, so the two white-noise mappings onto Basalt are the identity and only the two bias mappings carry any arithmetic at all.

### Ground truth sources

Every claim below is anchored to source rather than to documentation, because the Project AirSim documentation for this sensor is incomplete, as recorded under References.

| Concern | File | Lines |
|---|---|---|
| Parameter struct, SI-valued defaults, `min_sample_time` | `/ws/sitl_ws/src/ue5/ProjectAirSim/core_sim/include/core_sim/sensors/imu.hpp` | 33-52 |
| Settings parsing, no unit conversion, `bias_stability_norm` | `/ws/sitl_ws/src/ue5/ProjectAirSim/core_sim/src/sensors/imu.cpp` | 294-337 |
| Noise injection and bias propagation | `/ws/sitl_ws/src/ue5/ProjectAirSim/core_sim/src/sensors/imu.cpp` | 209-236 |
| Specific force, body-frame transform, unconditional noise | `/ws/sitl_ws/src/ue5/ProjectAirSim/core_sim/src/sensors/imu.cpp` | 174-207 |
| JSON key names | `/ws/sitl_ws/src/ue5/ProjectAirSim/core_sim/src/constant.hpp` | 254-260 |
| Gravity constant, 9.80665 m/s² | `/ws/sitl_ws/src/ue5/ProjectAirSim/core_sim/include/core_sim/earth_utils.hpp` | 72-73, 206-216 |
| Live configuration | `/ws/sitl_ws/config/airsim2/robot_ardu_copter.jsonc` | 345-366 |
| Simulation step, 3 ms | `/ws/sitl_ws/config/airsim2/scene_ardu_copter.jsonc` | `clock.step-ns` |
| Basalt discrete-time noise scaling | `thirdparty/basalt-headers/include/basalt/calibration/calibration.hpp` | 147-161 |
| Basalt bias random-walk residual | `include/basalt/linearization/imu_block.hpp` | 73-105 |

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

**Project AirSim field**: `gyroscope.angle-random-walk`, unit rad/s/√Hz, read directly with no conversion

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
parameter). This is related to an ARW quoted in the conventional units of °/√hr by:

    σ_θ²(T) = σ_c_g² × T   [rad² for T in seconds]

Setting T = 1 hr = 3600 s and converting ARW from °/√hr to rad/√hr:

    σ_c_g² × 3600 = (ARW × π/180)²

    σ_c_g = ARW × π / (180 × 60)   [rad/s / √Hz]

That derivation is the correct treatment of an ARW genuinely expressed in °/√hr, and it is the right formula for any data source that does use those units, for example a Kalibr report or a manufacturer datasheet. It is retained here because a configuration value obtained from such a source must pass through it before it can be written into the robot configuration.

#### No conversion is required and none is performed

`Imu::Loader::LoadGyroSettings` at `core_sim/src/sensors/imu.cpp:318-337` reads the JSON key `angle-random-walk` straight into `imu_settings.gyro.angle_random_walk` through `JsonUtils::GetNumber<float>`, with no arithmetic of any kind between the file and the struct member. The struct default at `core_sim/include/core_sim/sensors/imu.hpp:44-45` is written as `0.3 / sqrt(3600.f) * M_PI / 180` with the comment `deg/sqrt(hr) --> rad/sqrt(sec)`, so the conversion arithmetic above survives in Project AirSim only as the compile-time expression that produces the default. A configuration file must therefore supply the value already in rad/s/√Hz.

`ApplyNoiseModel` at `imu.cpp:228-229` then forms `imu_settings.gyro.angle_random_walk / sqrt_dt`, and division by √dt is the defining property of a continuous-time PSD, which confirms that the member is a continuous-time deviation and maps directly onto Basalt's `gyro_noise_std` with no scaling at all.

#### Basalt Equivalent

    Basalt field:   gyro_noise_std   [rad/s/√Hz = rad/√s]

    gyro_noise_std = gyroscope.angle-random-walk        (identity)

#### SITL Numerical Value

    angle-random-walk = 5.817764e-4      (configured, rad/s/√Hz)
    gyro_noise_std    = 5.8178e-4        rad/s/√Hz

#### How Basalt Uses It

`calibration.hpp` converts to a discrete-time standard deviation for pre-integration:

    σ_discrete_g = gyro_noise_std × √(imu_update_rate)   [rad/s]

This is the noise standard deviation per IMU sample used to build the pre-integration
covariance (see `preintegration.h:177-179`):

    Cov += G · diag(gyro_cov) · Gᵀ

where `gyro_cov = dicrete_time_gyro_noise_std².array()`.

---

### 1.2 Velocity Random Walk (VRW) — Accelerometer White Noise

**Project AirSim field**: `accelerometer.velocity-random-walk`, unit m/s²/√Hz, read directly with no conversion

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

As with the gyroscope, this is the correct treatment of a VRW genuinely expressed in m/s/√hr and the right formula for a datasheet or a Kalibr report quoting those units. It is not what the configuration file holds. A second convention in common use quotes VRW in milli-g, for which the conversion is a multiplication by 9.80665e-3, and the two differ by a factor of (1/60)/(9.80665e-3), which is 1.6995. Both conversions must be applied before the value enters the robot configuration, never after.

#### The field is SI despite its name

`Imu::Loader::LoadAccelerometerSettings` at `core_sim/src/sensors/imu.cpp:294-316` reads `velocity-random-walk` straight into `imu_settings.accelerometer.velocity_random_walk` with no arithmetic. The struct default at `core_sim/include/core_sim/sensors/imu.hpp:35-37` is written as `0.24f * gravity / 1.0E3f`, which is the milli-g conversion evaluated at compile time on a 0.24 mg datasheet figure, so once again the conversion survives only in the default. `ApplyNoiseModel` at `imu.cpp:216-217` forms `velocity_random_walk / sqrt_dt`, confirming the continuous-time reading.

The name is a trap. `velocity-random-walk` in this configuration is m/s²/√Hz, which is neither milli-g nor m/s/√hr, so neither of the two conventional conversions may be applied to the number in the file.

#### Unit Conversion

    σ_c_a [m/s²/√Hz]  =  velocity-random-walk                (Project AirSim, identity)
    σ_c_a [m/s²/√Hz]  =  VRW [mg]  ×  9.80665 × 10⁻³        (external source in milli-g)
    σ_c_a [m/s²/√Hz]  =  VRW [m/s/√hr]  ÷  60               (external source in m/s/√hr)

#### Basalt Equivalent

    Basalt field:   accel_noise_std   [m/s²/√Hz = m/s/√s]

    accel_noise_std = accelerometer.velocity-random-walk        (identity)

#### SITL Numerical Value

    velocity-random-walk = 1.176798e-2   (configured, m/s²/√Hz)
    accel_noise_std      = 1.1768e-2     m/s²/√Hz

#### How Basalt Uses It

Same pattern as gyro:

    σ_discrete_a = accel_noise_std × √(imu_update_rate)   [m/s²]

---

## Part 2 — Bias Parameters

### 2.1 The Two Bias Models Compared

The first-order Gauss-Markov theory in this subsection is correct general background and describes many real datasheets, but it is not what Project AirSim implements, and subsection 2.1b below records what the simulator actually does and governs wherever the two disagree.

#### First-Order Gauss-Markov (FOGM) Process, general background

The FOGM model treats bias as an Ornstein–Uhlenbeck process (mean-reverting random walk):

    ḃ(t) = -b(t)/τ  +  σ_w × w(t)

where:
- τ = the **time constant** of mean-reversion (seconds)
- σ_w = driving noise intensity [units/√s]
- w(t) = unit white noise
- Steady-state variance: E[b²] = σ_w² × τ/2  →  σ_b = σ_w × √(τ/2)

The parameter `bias-stability` (σ_b) is, under this reading, the **steady-state standard
deviation** of the bias, that is how large the bias gets on average over a very long time.

#### Basalt: Continuous-Time Random Walk

Basalt uses the simpler **Rate Random Walk** (RRW) model, also known as Wiener process:

    ḃ(t) ~ N(0, σ_rw²)   per unit time

Over interval Δt, bias evolves as: b(t+Δt) - b(t) ~ N(0, σ_rw² × Δt)

This is a limiting case of FOGM when Δt << τ (the bias drifts freely between FOGM
mean-reversion events). In this regime, the two models are equivalent with:

    σ_rw = σ_b × √(2/τ)   [units/√s]

This is the **conversion formula** between a genuine FOGM parameterisation and Basalt RRW parameters. It is NOT the conversion for Project AirSim, for the reason given in 2.1b, and applying it here overstates the result by exactly √2, which is 1.41421.

#### Relationship to Allan Deviation

In an Allan Deviation (ADEV) plot:
- **Bias Instability (BI)**: appears as the flat minimum region; corresponds to 1/f
  flicker noise. `bias-stability` ≈ BI value at the ADEV minimum.
- **Rate Random Walk (RRW)**: appears as the +½ slope region at longer averaging
  times; this is what Basalt `bias_std` models.
- **FOGM** approximates the transition between BI and RRW; τ marks the boundary.

The conversion `σ_rw = σ_b × √(2/τ)` comes from matching the FOGM PSD to a
two-sided approximation of the 1/f + RRW spectrum.

### 2.1b What Project AirSim Actually Implements

Project AirSim propagates the bias as a pure Wiener process, that is a plain random walk with no mean reversion at all, which is exactly Basalt's rate random walk model. The two therefore agree with no approximation and the conversion is a simple division.

The normalised coefficient is precomputed at load time, at `core_sim/src/sensors/imu.cpp:311-313` and `:333-334`.

```cpp
impl.accelerometer_bias_stability_norm =
    impl.imu_settings.accelerometer.bias_stability /
    sqrt(impl.imu_settings.accelerometer.tau);
...
impl.gyroscope_bias_stability_norm =
    impl.imu_settings.gyro.bias_stability / sqrt(impl.imu_settings.gyro.tau);
```

The propagation at `imu.cpp:222-224` and `:234-235` is `sigma_bias = bias_stability_norm * sqrt_dt` followed by `state_.bias += gauss.next() * sigma_bias`, with no −b/τ term anywhere in the update. The increment over an interval dt therefore has standard deviation (σ_b/√τ)·√dt, and the conversion to Basalt is the following.

    bias_std  =  bias-stability / √tau

Because `bias-stability` is already in rad/s for the gyroscope and m/s² for the accelerometer, no prior unit conversion applies and this division is the whole of the mapping.

A consequence worth recording is that because the simulated bias never mean-reverts, the simulated device does not exhibit the flat bias-instability shelf that the parameter name `bias-stability` implies. The parameter serves only to set the random walk coefficient in combination with tau.

Three further consequences of the load-time initialisation deserve recording, because none is visible from the configuration file. First, `accelerometer_bias_stability_norm` and `gyroscope_bias_stability_norm` are declared at `imu.cpp:95` with no in-class initialiser and are assigned only inside `LoadAccelerometerSettings` and `LoadGyroSettings`. `LoadImuSettings` at `:270-292` skips those loaders entirely when the corresponding JSON object is absent or empty, in which case the coefficient is read uninitialised on the first noise update. A robot configuration must therefore supply both an `accelerometer` and a `gyroscope` object even if it only wishes to accept the defaults, and `robot_ardu_copter.jsonc:354-365` does. Second, `turn-on-bias` seeds `state_.accelerometer_bias` and `state_.gyroscope_bias` directly, so it is a deterministic constant offset added to every sample and not a distribution to be sampled from. It is zero here, which is what makes `calib_accel_bias` and `calib_gyro_bias` correctly all-zero, as recorded in [`imu_static_calibration.md`](imu_static_calibration.md). Third, the `accelerometer.gravity` key is loadable at `:296-297` but is never read again anywhere in the noise model, which uses `ground_truth_.environment->env_info.gravity` instead at `:187-190`. Setting it has no effect beyond documenting the constant the defaults were computed against.

---

### 2.2 Gyroscope Bias — `gyroscope.bias-stability` and `gyroscope.tau`

**Project AirSim fields**:
- `gyroscope.bias-stability` — random walk coefficient numerator, unit: **rad/s**, read directly
- `gyroscope.tau` — time constant, unit: **seconds**, read directly

#### Unit Conversion

`core_sim/src/sensors/imu.cpp:325-327` reads `bias-stability` into a member already denominated in rad/s, so σ_b_g is the configured number itself and no conversion applies. A value taken from a datasheet quoting °/hr must be converted before it enters the file:

    σ_b_g [rad/s]  =  bias stability [°/hr]  ×  π / (180 × 3600)
                    =  bias stability [°/hr]  ×  4.848 × 10⁻⁶

Conversion to the Basalt random walk model is then:

    gyro_bias_std [rad/s/√s]  =  σ_b_g / √tau

#### Basalt Equivalent

    Basalt field:   gyro_bias_std   [rad/s/√s = rad/s²/√Hz]

#### SITL Numerical Value

    bias-stability = 9.696274e-5 rad/s   (configured)
    tau            = 300 s               (configured)

    gyro_bias_std = 9.696274e-5 / √300 = 5.5981e-6  rad/s/√s

This is the value the simulated sensor alone justifies. It is not the value that fits the stream Basalt receives, because ArduPilot superimposes a second and larger bias process of its own that no simulator setting describes. Part 7.3 derives it and states the recommended figure.

#### How Basalt Uses It

In `sqrt_keypoint_vio.cpp:116`:

    gyro_bias_sqrt_weight = calib.gyro_bias_std.array().inverse()

This is used as the information weight (1/σ) on the bias prior/regularization in the
factor graph. A smaller `gyro_bias_std` means Basalt trusts the bias to stay fixed
— tighter regularization. Conversely an inflated value permits the bias state to absorb
error that properly belongs to another part of the model.

---

### 2.3 Accelerometer Bias — `accelerometer.bias-stability` and `accelerometer.tau`

**Project AirSim fields**:
- `accelerometer.bias-stability` — random walk coefficient numerator, unit: **m/s²**, read directly
- `accelerometer.tau` — time constant, unit: **seconds**, read directly

#### Unit Conversion

`imu.cpp:303-305` reads `bias-stability` into a member already denominated in m/s², so no conversion applies. A value taken from a datasheet quoting micro-g must be converted before it enters the file, using the gravity constant 9.80665 exactly, per `core_sim/include/core_sim/earth_utils.hpp:72-73`, rather than the 9.81 sometimes used:

    σ_b_a [m/s²]  =  bias stability [μg]  ×  9.80665 × 10⁻⁶

Conversion to the Basalt random walk model is then:

    accel_bias_std [m/s²/√s]  =  σ_b_a / √tau

#### Basalt Equivalent

    Basalt field:   accel_bias_std   [m/s²/√s]

    Note: the unit is written as m/s²/√s to emphasize it is a continuous-time
    rate-of-change noise; per unit root-second of elapsed time, the bias drifts
    by accel_bias_std m/s².

#### SITL Numerical Value

    bias-stability = 4.903325e-3 m/s²    (configured)
    tau            = 600 s               (configured)

    accel_bias_std = 4.903325e-3 / √600 = 2.0018e-4  m/s²/√s

Unlike the gyroscope, this figure needs no inflation for the ArduPilot path. `AP_InertialSensor_SITL::generate_accel` adds only the constant `SIM_ACC1_BIAS`, which defaults to zero and is not set here, and no accelerometer counterpart of the gyroscope drift ramp exists. See Part 7.3.

#### How Basalt Uses It

Same as gyro_bias_std — used as the inverse weight on the bias prior:

    accel_bias_sqrt_weight = calib.accel_bias_std.array().inverse()

---

## Part 3 — IMU Update Rate

**Project AirSim**: The sensor is updated once per scene tick, since `Imu::Impl::Update` at `core_sim/src/sensors/imu.cpp:174-207` takes `sim_dt_nanos` from the scene and the comment at `:179-181` states that this is the fastest ticking component. With `clock.step-ns` of 3000000 in `scene_ardu_copter.jsonc` that is 3 ms, so the simulated sensor produces 333.3 samples per second of simulation time and the bias random walk advances once per sample. There is no per-sensor rate key, so the only way to change it is to change the scene step. `min_sample_time` survives at `core_sim/include/core_sim/sensors/imu.hpp:51` with a value of 1/1000 and is not loadable from JSON. It is a floor under `sqrt_dt` at `imu.cpp:211-212`, so a scene step below 1 ms would silently over-inject noise by the factor √(0.001/step); at 3 ms the clamp is inactive and the continuous-time reading is exact.

**Basalt field**: `imu_update_rate` [Hz]

This scalar is used only in the discrete-time conversion, implemented at
`thirdparty/basalt-headers/include/basalt/calibration/calibration.hpp:147-161`:

    σ_discrete = σ_continuous × √(imu_update_rate)

It does NOT affect the underlying physical noise parameters — only how Basalt maps
the continuous-time PSD to per-sample noise. If the IMU rate changes, only this
field needs updating; the noise std fields remain the same.

> **The field must be set from the rate at the subscriber, not from the sensor.** Neither the 333.3 Hz scene tick nor the nominal 1000 Hz internal figure is the rate at which Basalt receives distinct samples. Basalt subscribes to `/ap/imu/experimental/data`, which AP_DDS gates at `AP_DDS_DELAY_IMU_TOPIC_MS` of 5 with a strict inequality at `libraries/AP_DDS/AP_DDS_Client.cpp:1930-1935`, giving an inertial period of exactly 6 ms and a rate of 166.667 Hz. Every stamp on that topic is distinct, so the distinct-sample rate equals the message rate and the correct entry is 167. The measurement, the mechanism and the proof that no stamp can repeat are in [`ap_dds_imu_stream.md`](ap_dds_imu_stream.md).
>
> One property of this field is easy to get wrong and worth stating explicitly. It is a rate in the time domain of the header stamps, not in wall-clock time. Under `use_sim_time` those stamps are simulation time, so `imu_update_rate` must be the simulation-time rate and is invariant to the ratio of `step-ns` to `real-time-update-rate` that scales the simulation against the wall. A stack running at one third real time still requires 167, not 55.7.

---

## Part 4 — The Noise Cannot Be Disabled

Project AirSim has no flag that gates the noise model. `Imu::Impl::Update` calls `ApplyNoiseModel` unconditionally at `core_sim/src/sensors/imu.cpp:195`, there is no `add_noise` member on `ImuParams` and no gate anywhere in `ApplyNoiseModel` at `:209-236`. A clean IMU is obtained only by setting all four noise parameters and `turn-on-bias` to zero explicitly.

The practical consequence runs in two directions. A Project AirSim configuration can never be accidentally noiseless, so a SLAM result can never be flattered by a silently disabled noise model, and conversely the four parameters are always live and their correctness always matters.

Implications for SLAM:
- Because noise is always injected, VIO performance always reflects the configured densities, and coarse Basalt IMU parameters will show.
- Real-world deployment requires re-calibrating all four noise parameters against the physical device.
- For testing IMU integration correctness, scale the four parameters up and verify SLAM degrades gracefully as noise increases, then restore them.

---

## Part 5 — Complete Conversion Table

### Configured Project AirSim values

Read from `/ws/sitl_ws/config/airsim2/robot_ardu_copter.jsonc:345-366`. Every value is already SI and every white-noise conversion is the identity, so only the two bias rows carry any arithmetic at all.

| Project AirSim key | Value | Unit | Basalt Field | Basalt Value | Basalt Unit | Conversion |
|---|---|---|---|---|---|---|
| `gyroscope.angle-random-walk` | 5.817764e-4 | rad/s/√Hz | `gyro_noise_std` | 5.8178e-4 | rad/s/√Hz | identity |
| `accelerometer.velocity-random-walk` | 1.176798e-2 | m/s²/√Hz | `accel_noise_std` | 1.1768e-2 | m/s²/√Hz | identity |
| `gyroscope.bias-stability` + `tau` | 9.696274e-5 + 300 | rad/s + s | `gyro_bias_std` | 5.5981e-6 | rad/s/√s | σ_b / √τ |
| `accelerometer.bias-stability` + `tau` | 4.903325e-3 + 600 | m/s² + s | `accel_bias_std` | 2.0018e-4 | m/s²/√s | σ_b / √τ |

The device modelled is an MPU-6000, considerably noisier than the struct defaults at `core_sim/include/core_sim/sensors/imu.hpp:33-52`. For reference those defaults correspond to 0.3 °/√hr, 4.6 °/hr, 0.24 mg and 36 μg, giving 8.726646e-5, 2.230143e-5, 2.353596e-3 and 3.530394e-4 respectively under the conversions of Parts 1 and 2, and Project AirSim's own example configurations reproduce those figures to float precision. The configured values here are roughly a factor of five noisier on every axis.

### Discrepancy against the values currently in data/sitl_calib.json

| Basalt field | In `sitl_calib.json` | Correct | Ratio | Verdict |
|---|---|---|---|---|
| `gyro_noise_std` | 5.818e-4 | 5.8178e-4 | 1.0000 | correct |
| `accel_noise_std` | 2.000e-2 | 1.1768e-2 | 1.6995 | too large, the ÷60 formula applied to a milli-g figure |
| `gyro_bias_std` | 7.92e-6 | 5.5981e-6 | 1.4148 | too large by √2, the spurious FOGM factor of 2.1 |
| `accel_bias_std` | 2.83e-4 | 2.0018e-4 | 1.4137 | too large by √2, the spurious FOGM factor of 2.1 |
| `imu_update_rate` | 167 | 167 | 1.0000 | correct |

Three of the four noise values are wrong and all three errors are overestimates, which is the conservative direction. An overestimate causes the estimator to under-trust the IMU rather than to diverge, so these errors degrade accuracy but they do not by themselves explain a diverging trajectory under stationary conditions. The accelerometer density is the largest of the three and is the one to correct first. `imu_update_rate` is correct, and because no inertial sample is lost between the publisher and the estimator there is no coupling between that field and a loss of data, so the per-sample noise scaling it drives is faithful.

A caveat on `imu_update_rate` that follows from the calibration file being shared. `djinn:366` and `djinn:409` both pass `sitl_calib.json`, so more than one launch branch reads one file. The four noise values are properties of the sensor model and are correct for any branch that runs this robot configuration, but 167 is a property of the AP_DDS gate and of the stamp source, so a branch whose inertial topic delivers a different distinct-sample rate would need its own value.

### Resulting Basalt Calibration JSON Block

```json
"imu_update_rate": 167,
"gyro_noise_std":  [5.8178e-4, 5.8178e-4, 5.8178e-4],
"accel_noise_std": [1.1768e-2, 1.1768e-2, 1.1768e-2],
"gyro_bias_std":   [1.5586e-5, 1.5586e-5, 1.5586e-5],
"accel_bias_std":  [2.0018e-4, 2.0018e-4, 2.0018e-4]
```

Three of the four entries are the simulator values derived above. `gyro_bias_std` is not, and the reason is Part 7.3: ArduPilot superimposes a deterministic gyroscope drift that the simulator configuration knows nothing about and that dominates the configured random walk by a factor of twenty over a minute. The figure quoted is the quadrature sum of the simulator's 5.5981e-6 with the 1.4544e-5 rad/s/√s that makes the ArduPilot ramp a one-sigma event over a one second window. If instead `SIM_DRIFT_SPEED` is set to zero in the ArduPilot parameter file, the drift disappears and the correct entry reverts to the derived 5.5981e-6, which is the preferable arrangement because it leaves exactly one bias process in the stack and that process is the one the calibration describes.

### Comparison: EuRoC vs SITL

| Parameter | EuRoC ADIS16448 | SITL | Ratio |
|---|---|---|---|
| `gyro_noise_std` (rad/s/√Hz) | 2.82e-4 | 5.82e-4 | 2.1× noisier |
| `accel_noise_std` (m/s²/√Hz) | 1.60e-2 | 1.18e-2 | 1.4× quieter |
| `gyro_bias_std` (rad/s/√s) | 1.00e-4 | 5.60e-6 | 18× more stable |
| `accel_bias_std` (m/s²/√s) | 1.00e-3 | 2.00e-4 | 5× more stable |

The configured simulation is broadly comparable to the EuRoC sensor on white noise and markedly more stable on bias, so the IMU model itself is not pathological and the IMU term retains meaningful weight in the factor graph. Note that this conclusion depends on using the values derived here. A calibration built from the erroneous entries of the discrepancy table, or from the struct defaults, would make the simulated IMU appear between three and seventy times better than the EuRoC sensor and would wrongly suggest that the VIO should rely on vision and lean lightly on IMU prediction between frames.

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
             tau (where FOGM transitions to RRW)
```

| ADEV Region | Slope | Project AirSim Parameter | Basalt Parameter |
|---|---|---|---|
| Short τ, slope −½ | -0.5 | `angle-random-walk` / `velocity-random-walk` | `gyro_noise_std` / `accel_noise_std` |
| Middle τ, flat | 0 | `bias-stability` | (approximated via conversion) |
| Long τ, slope +½ | +0.5 | `bias-stability` + `tau` (FOGM tail) | `gyro_bias_std` / `accel_bias_std` |

This plot describes a real device. As established in 2.1b, the Project AirSim simulated device has no mean-reversion and therefore no flat bias-instability shelf, so its own ADEV would show only the −½ and +½ slopes meeting directly.

---

## Part 7 — What Actually Reaches Basalt

Every figure derived so far describes the simulated sensor. Basalt does not subscribe to the simulated sensor. It subscribes to `/ap/imu/experimental/data`, an ArduPilot topic, and three transformations stand between the two that no simulator setting describes. This part enumerates them, quantifies each, and states which of them the calibration must absorb.

The chain is as follows, with each hop anchored to source.

| Stage | Where | Effect |
|---|---|---|
| Simulated IMU sample, noise and bias injected | `core_sim/src/sensors/imu.cpp:174-236` | 333.3 Hz sim time, body NED, σ_c as configured |
| Serialised into the ArduPilot JSON packet, verbatim | `vehicle_apis/multirotor_api/src/arducopter_api.cpp:705-714` | no conversion, no rescaling |
| Parsed into the SITL flight dynamics model | `libraries/SITL/SIM_JSON.cpp:383-384` then `SIM_Aircraft.cpp:410-415` | becomes `sitl->state.xAccel` and `rollRate` |
| ArduPilot's own inertial backend adds noise, drift and bias | `libraries/AP_InertialSensor/AP_InertialSensor_SITL.cpp:114-120, 218-249` | 1000 Hz, additive |
| Low-pass filtered at `INS_ACCEL_FILTER` and `INS_GYRO_FILTER` | `Tools/autotest/default_params/airsim-quadX.parm`, both 50 Hz | attenuates above 50 Hz |
| Published by AP_DDS every 6 ms of simulation time | `libraries/AP_DDS/AP_DDS_Client.cpp:1930-1935` | 166.7 Hz, every stamp distinct |

### 7.1 ArduPilot's additive white noise is negligible

`generate_accel` adds one uniform draw per sample at `AP_InertialSensor_SITL.cpp:120`, with amplitude fixed at 0.01 m/s² and not parameterised. `SIM_ACC1_RND` updates the amplitude only for the vibration terms further down, which are inert here because `SIM_VIB_FREQ` and `SIM_VIB_MOT_MAX` both default to zero. `generate_gyro` adds two independent draws, at `:230-232` and again at `:247-249`, the second because the vibration branch at `:245` is taken precisely when no vibration is configured, with amplitude `radians(0.04)` and likewise not parameterised in the motors-off case. `rand_float` is uniform on [−1, 1) per `libraries/AP_Math/AP_Math.cpp:361-368`, so its standard deviation is 1/√3.

    accel  σ_d = 0.01 / √3            = 5.7735e-3 m/s²  per sample at 1000 Hz
           σ_c = 5.7735e-3 / √1000    = 1.8257e-4 m/s²/√Hz
    gyro   σ_d = radians(0.04)·√(2/3) = 5.7002e-4 rad/s per sample at 1000 Hz
           σ_c = 5.7002e-4 / √1000    = 1.8026e-5 rad/s/√Hz

Added in quadrature to the simulator's own densities this raises `accel_noise_std` by a factor of 1.0001 and `gyro_noise_std` by 1.0005. Both are far inside the precision to which the parameters are known and neither justifies changing the calibration. The conclusion to carry forward is that the simulator configuration, not ArduPilot, sets the white noise seen by Basalt.

### 7.2 The 50 Hz filter changes the visible scatter but not the preintegration covariance

`INS_ACCEL_FILTER` and `INS_GYRO_FILTER` are both 50 in `airsim-quadX.parm`, against a hard-coded default of 20, so a single-pole low pass at 50 Hz sits between the 1000 Hz sample generation and `AP::ins().get_accel()`. Passing white noise of density σ_c through it gives an output variance of σ_c²·(π/2)·f_c, which is 0.1043 m/s² for the accelerometer and 5.156e-3 rad/s for the gyroscope. Basalt, forming σ_d = σ_c·√167 at `calibration.hpp:147-161`, assumes 0.1521 and 7.518e-3 instead. The per-sample scatter it expects is therefore 1.46 times what a `ros2 topic echo` would show, and a reader who measures the sample standard deviation directly and back-solves for σ_c will obtain 8.07e-3 rather than 1.1768e-2 and conclude, wrongly, that the calibration is too large.

The calibration is not too large, and the reason is that preintegration integrates the noise rather than sampling it. A low pass with unity DC gain leaves the low-frequency power spectral density untouched, and the variance of the integral of the noise over an interval T is σ_c²·T for any T long compared with the filter's correlation time of 1/(2π·50), which is 3.2 ms. Preintegration intervals between keyframes are at least an order of magnitude longer than that, so the accumulated covariance is the one the unfiltered density predicts. The correct entry is σ_c, and the filter must not be compensated for.

### 7.3 ArduPilot's gyroscope drift is the one contribution that does matter

`AP_InertialSensor_SITL::gyro_drift` at `:401-413` adds a deterministic triangular ramp, identical on all three axes, governed by `SIM_DRIFT_SPEED` and `SIM_DRIFT_TIME`, which default to 0.05 °/s per minute and 5 minutes at `libraries/SITL/SITL.cpp:73` and `:77` and are not overridden by any parameter file in this stack. The ramp rate is radians(0.05)/60, which is 1.4544e-5 rad/s per second, and it reverses every five minutes at a peak of 4.3633e-3 rad/s, or 0.25 °/s.

Basalt's bias random-walk factor penalises the difference between consecutive bias states by `gyro_bias_std·√dt`, at `include/basalt/linearization/imu_block.hpp:73-85`, so the question is how the deterministic ramp compares against the one-sigma excursion that prior permits.

| Window | ArduPilot ramp | 1σ permitted by 5.5981e-6 | Ratio |
|---|---|---|---|
| 0.1 s | 1.454e-6 | 1.770e-6 | 0.82 |
| 1 s | 1.454e-5 | 5.598e-6 | 2.60 |
| 10 s | 1.454e-4 | 1.770e-5 | 8.22 |
| 60 s | 8.727e-4 | 4.336e-5 | 20.12 |

The crossover is at 0.148 s. Below it the prior is adequate; above it the prior fights a real and monotonic drift, and the ratio grows without bound because a ramp accumulates linearly while a random walk accumulates as the square root. The estimator does not diverge, since the inertial residuals themselves push the bias state, but the prior lags systematically, and a lagging gyroscope bias tilts the estimated attitude, and an attitude error θ leaks a horizontal specific force of approximately 9.81·sin θ into the estimate, which double integrates into position.

Two remedies exist and they are not equivalent. Setting `SIM_DRIFT_SPEED` to zero removes the second bias process outright, leaving the simulator's own random walk as the only one and making the derived `gyro_bias_std` of 5.5981e-6 exact by construction. Inflating `gyro_bias_std` to cover the ramp keeps the drift but models it as a random walk it is not, and requires choosing the window over which the two are to be matched; matching at one second gives 1.4544e-5, which in quadrature with the simulator's own figure is 1.5586e-5. The first is preferable because it eliminates a mis-specification rather than accommodating it, and because it leaves the calibration derivable from the robot configuration alone.

There is no accelerometer counterpart. `generate_accel` adds only `SIM_ACC1_BIAS`, a constant that defaults to zero, so `accel_bias_std` needs no adjustment.

### 7.4 Two smaller effects, recorded for completeness

The inertial stamp and the image stamp are produced by different mechanisms. An inertial sample carries ArduPilot's stamp, which on the external-clock branch is quantised to the `/clock` period and on the fallback branch is the simulated RTC read at microsecond resolution, while an image carries the simulator's own per-sample `time_stamp`, exact to the nanosecond. On the quantised branch the inertial stream appears systematically early relative to the camera by half a quantum, and working against it the 50 Hz filter delays the inertial payload by roughly 1/(2π·50), which is 3.2 ms, so the two errors partially cancel and the residual is of order a millisecond. A far larger hazard exists on the fallback branch, where the two streams occupy different epochs entirely, and it is treated in [`ap_dds_imu_stream.md`](ap_dds_imu_stream.md). Basalt's `cam_time_offset_ns` cannot be used to compensate either effect, because the line that applies it is commented out at `src/vi_estimator/sqrt_keypoint_vio.cpp:231`; the field is honoured only by the offline calibration tools.

Gravity is altitude-dependent. `EarthUtils::GetGravity` at `core_sim/include/core_sim/earth_utils.hpp:206-216` returns 9.80665·(1 − 2·h/R) above sea level, giving 9.8048 at the 583 m home altitude of `scene_ardu_copter.jsonc`, against the 9.81 that Basalt hard-codes at `include/basalt/utils/imu_types.h:62`. The 5.2e-3 m/s² difference is a constant that the accelerometer bias state absorbs, and it is a factor of forty below the 0.2149 m/s² per-sample noise, so it is recorded rather than acted upon.

---

## Related context

- [`airsim_camera_extrinsics.md`](airsim_camera_extrinsics.md), the T_imu_cam derivation and the NED body frame this document's IMU shares.
- [`ap_dds_imu_stream.md`](ap_dds_imu_stream.md), the transport, frame, rate and timestamp behaviour of the live IMU topic.
- [`imu_static_calibration.md`](imu_static_calibration.md), the deterministic `calib_accel_bias` and `calib_gyro_bias` intrinsics, which are distinct from the stochastic parameters here.

## References

- IEEE Std 952-1997, "IEEE Standard Specification Format Guide and Test Procedure for
  Single-Axis Interferometric Fiber Optic Gyros" — defines ARW, BI, RRW in Allan Variance.
- Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation", TR 2005-002
  — continuous vs discrete IMU noise models.
- Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry",
  IEEE T-RO 2017 — IMU preintegration covariance with continuous-time noise model.
- Usenko et al., "Visual-Inertial Mapping with Non-Linear Factor Recovery",
  RA-L 2020 (arXiv:1904.06504) — Basalt's factor structure and IMU weighting.
- Project AirSim IMU implementation: `core_sim/include/core_sim/sensors/imu.hpp` and
  `core_sim/src/sensors/imu.cpp` — the authoritative source, and the only source, since
  `docs/config_robot.md:712` links an "IMU Settings" page that does not exist in the tree
  and `imu.hpp:31` points at a `docs/sensors/imu.md` that is likewise absent.
- ArduPilot simulated inertial backend: `libraries/AP_InertialSensor/AP_InertialSensor_SITL.cpp`
  — the additive noise, the drift ramp and the filter that stand between the simulator
  and the topic Basalt consumes.
- Kalibr IMU noise model wiki — the standard reference for the ARW, VRW and bias random
  walk parameterisation these fields follow.
