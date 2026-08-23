# VIO ↔ Local Mapper Correction Loop Analysis

**Date**: 2026-07-11
**Snapshots analysed**: `/ws/gt-drift.png`, `/ws/gt-drift-2.png`, `/ws/gt-drift-3.png`

---

## Observed Symptoms

1. **Constant offset between GT and SLAM trajectories** — both trajectories follow the same shape (correct alignment fix applied) but diverge in position over time.
2. **Offset between SLAM trajectory and local mapper KF frusta** — the pink/amber local mapper keyframe frustums appear displaced relative to the red VIO trajectory line.

---

## Visualiser Color Map

| Element | Color constant | RGB | Description |
|---------|---------------|-----|-------------|
| VIO trajectory line | `cam_color` | (250, 0, 26) red | `mvpVioTrajectory` from `out_state_queue` |
| VIO sliding-window states | `state_color` | (250, 0, 26) red | `vio->states` frusta |
| VIO non-KF frames | `pose_color` | (0, 50, 255) blue | `vio->frames` frusta |
| GT trajectory | `gt_color` | (0, 171, 47) green | `mvpGroundTruthTrajectory` |
| Local map points | `local_map_point_color` | (255, 128, 0) orange | 3D landmarks from BA |
| Local map KF frusta | `local_map_kf_color` | (255, 128, 128) pink/amber | `lm->keyframes` BA-refined poses |

---

## Correction Data Flow (as-built)

```
VIO (SqrtKeypointVioEstimator)
│
├── frame_states   ← active sliding window poses (IMU-integrated)
│       │
│       └── out_state_queue ──► GUI mvpVioTrajectory (red line)
│                                    ▲ NEVER CORRECTED by local mapper
│
├── frame_poses    ← marginalized KFs (pose-only, frozen from VIO window)
│       │
│       ├── out_marg_queue ──────────────────────────────────────────►
│       │                                                              │
│       └── mpPosesToUpdate ◄── QueuePoseUpdates ◄── PoseUpdateCallback
│                 │
│                 └── applied in measure() (next VIO frame) to frame_poses only
│
LocalMapper (on separate thread)
│
├── frame_poses ← seeded from MargData (VIO's frame_poses at marginalization time)
│       │
│       └── optimize() ► BA-refined poses ──► out_vis_queue ──► GUI pink frusta
│                 │
│                 └── PoseUpdateCallback(frame_poses) ──► VIO QueuePoseUpdates
│
└── lmdb        ← landmarks (world-frame, consistent with LocalMapper frame_poses)
```

**Critical observation**: The live VIO trajectory (`frame_states` → `out_state_queue`) is in a separate data path from the correction loop. BA corrections from the local mapper can only reach VIO's `frame_poses` (the frozen, already-marginalized set), never `frame_states`. The GUI trajectory is driven by `frame_states` exclusively.

---

## Issue 1 — GT vs SLAM Offset

**Root cause**: VIO drift, not alignment error.

The two-part GT alignment fix (IO normalization + `mpSlamFirstPose` visualiser correction) correctly aligns the origins. The divergence observed in the screenshots is accumulated VIO position error from the sliding-window estimator running without any global correction.

This is unavoidable without loop closure. The local mapper runs local BA but:
- Its corrections only reach marginalized `frame_poses`, not the live trajectory
- It has no loop closure (no place recognition for revisited scenes)
- Without loop closure, translation drift grows unboundedly over time

---

## Issue 2 — SLAM Trajectory vs Local Mapper KF Offset

**Three mechanisms** cause the pink KF frusta to appear displaced from the red VIO trajectory:

### Mechanism A — No live-trajectory correction (architectural)

The VIO trajectory (red line in GUI) is `frame_states` → `out_state_queue`. The local mapper BA refines its own copy of `frame_poses`. The feedback callback path is:

```
LocalMapper::MapLocally()
  → mpVioPoseUpdateCallback(frame_poses)   [local_mapper.cpp:246]
  → Controller lambda
  → vio_estimator_->QueuePoseUpdates(poses)
  → mpPosesToUpdate staging map           [sqrt_keypoint_vio.h:219]
  → measure() applies to VIO frame_poses  [sqrt_keypoint_vio.cpp:364-398]
```

`frame_states` is never touched by this path. The live trajectory that the GUI renders does not receive any correction from the local mapper.

### Mechanism B — Hard correction threshold blocks large updates

From `sqrt_keypoint_vio.h:290`:
```cpp
static constexpr Scalar kRelinThresholdTrans = Scalar(0.10);  // 10 cm
static constexpr Scalar kRelinThresholdRot   = Scalar(0.05);  // ~3 degrees
```

In `measure()` (lines 378-384), corrections where the BA-refined pose differs from the VIO pose by more than 10 cm or 3° are silently rejected with `"too large update in pose, not updating"`.

As VIO drift accumulates, the gap between VIO's `frame_poses` and the BA-refined `frame_poses` grows. The threshold means corrections become progressively less effective as drift increases — exactly when correction is most needed.

### Mechanism C — Translational gauge freedom in NfrMapper BA

`NfrMapper::optimize()` (`nfr_mapper.cpp:249`) only has:
- `rel_pose_factors` — constrain **relative** poses between KFs (shape of trajectory, not position)
- `roll_pitch_factors` — constrain gravity alignment (2 DOF: roll + pitch)

There is **no absolute position constraint** in the local BA problem. The remaining 4 DOF (absolute translation XYZ + yaw) are unconstrained. If reprojection error is dominant and pulls poses off-axis, the BA solution can be self-consistent but translated/rotated relative to the VIO world frame. This causes the local mapper KFs to appear shifted from where VIO placed them.

This is the "gauge freedom" problem: `rel_pose_factors` fix the SHAPE of the keyframe constellation; `roll_pitch_factors` fix the gravity direction; but the LOCATION of the constellation in world space is unconstrained by the factor graph alone. The only implicit anchor is the initial values of `frame_poses` (seeded from VIO), which acts as a prior, but it is not explicitly enforced.

---

## What the Local Mapper BA DOES Achieve

Despite not correcting the live trajectory, local BA provides:
1. **More accurate KF poses** — refined by visual reprojection over cross-frame tracks
2. **Better landmark positions** — triangulated with refined poses, so the map is geometrically consistent
3. **Improved future frame tracking** — if VIO re-encounters these KFs, the better map geometry reduces reprojection error

What it CANNOT do in the current architecture:
- Correct the running VIO trajectory (`frame_states`)
- Apply corrections larger than 10 cm / 3° back to VIO's `frame_poses`
- Perform loop closure (no place recognition for revisited scenes)

---

## Path to Fixing the Gap

To make local mapper corrections propagate to the live VIO trajectory:

**Option 1 — Sliding window reanchoring**: When the local mapper refines a KF that is still in the VIO window (unlikely given the one-way marg flow, but possible if the window is large), apply the correction as a pose prior in the VIO optimization.

**Option 2 — Velocity/position injection**: Compute the delta between VIO's current `frame_states` origin and the BA-corrected origin. Apply this delta as a velocity or position jump at the start of the next IMU preintegration. Risks consistency violation (FEJ breakage).

**Option 3 — Loop closure with pose graph correction**: Detect revisited places. When a loop is closed, apply a global SE(3) correction to ALL states (past and present). This is the standard ORB-SLAM3 / VINS-Fusion approach.

**Option 4 — Fix gauge freedom in NfrMapper**: Add an absolute position prior on the oldest KF (fix its pose to the VIO-provided value with tight covariance). This prevents the BA from floating in translation, so corrections fed back to VIO are in the same world frame.

---

## Relevant Code Locations

| Location | Role |
|----------|------|
| `src/vi_estimator/sqrt_keypoint_vio.cpp:364-398` | Where corrections are consumed; threshold rejection |
| `src/vi_estimator/sqrt_keypoint_vio.h:215-220` | `QueuePoseUpdates` — writes staging map |
| `src/vi_estimator/sqrt_keypoint_vio.h:281-292` | `mpPosesToUpdate`, `kRelinThresholdTrans/Rot` |
| `src/vi_estimator/local_mapper.cpp:246` | Where local mapper fires callback after BA |
| `src/controller.cpp:167-168` | Wires the callback: `LocalMapper → QueuePoseUpdates` |
| `src/vi_estimator/nfr_mapper.cpp:249-390` | BA implementation (no absolute position prior) |
| `src/visualisation/visualiser.cpp:260-300` | DrawScene: trajectory + KF frusta rendering |

## The `out_state_queue` tap became non-blocking, 2026-08-23

The push at `src/vi_estimator/sqrt_keypoint_vio.cpp:594` that feeds `mvpVioTrajectory` is now `try_push` rather than `push`, honouring the contract stated at `src/visualisation/visualiser.cpp:36-37`. The queue holds one hundred states and the consumer appends rather than replaces, so a drop is a permanent gap in the red trajectory line and a missing row in the plotter log rather than a stale value. The cost is cosmetic, since neither structure feeds the estimate, and it buys the guarantee that a stalled Pangolin thread can no longer block the estimator. The first state cannot be dropped, because the queue is necessarily empty when it is pushed, which matters since `ConsumeVioStateQueue` captures `mpSlamFirstPose` from it for ground truth alignment.

The companion tap at `:621` feeding `mvpVioVisQueue` is also `try_push` now. That queue is already latest wins, since `ConsumeVioVisQueue` stores into `mpLatestVio` and `Run` reads only that pointer, so a drop there is invisible beyond a frame of staleness.

Full analysis in `/ws/ros_ws/src/slam/plans/basalt-local-mapper-deadlock.md` section 4.6.
