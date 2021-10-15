/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2021, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/linearization/linearization_abs_qr.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <basalt/utils/ba_utils.h>
#include <basalt/linearization/imu_block.hpp>
#include <basalt/utils/cast_utils.hpp>

namespace basalt {

template <typename Scalar, int POSE_SIZE>
LinearizationAbsQR<Scalar, POSE_SIZE>::LinearizationAbsQR(
    BundleAdjustmentBase<Scalar>* estimator, const AbsOrderMap& aom,
    const Options& options, const MargLinData<Scalar>* marg_lin_data,
    const ImuLinData<Scalar>* imu_lin_data,
    const std::set<FrameId>* used_frames,
    const std::unordered_set<KeypointId>* lost_landmarks,
    int64_t last_state_to_marg)
    : options_(options),
      estimator(estimator),
      lmdb_(estimator->lmdb),
      frame_poses(estimator->frame_poses),
      calib(estimator->calib),
      aom(aom),
      used_frames(used_frames),
      marg_lin_data(marg_lin_data),
      imu_lin_data(imu_lin_data),
      pose_damping_diagonal(0),
      pose_damping_diagonal_sqrt(0) {
  UNUSED(last_state_to_marg);

  BASALT_ASSERT_STREAM(
      options.lb_options.huber_parameter == estimator->huber_thresh,
      "Huber threshold should be set to the same value");

  BASALT_ASSERT_STREAM(options.lb_options.obs_std_dev == estimator->obs_std_dev,
                       "obs_std_dev should be set to the same value");

  // Allocate memory for relative pose linearization
  for (const auto& [tcid_h, target_map] : lmdb_.getObservations()) {
    // if (used_frames && used_frames->count(tcid_h.frame_id) == 0) continue;
    const size_t host_idx = host_to_idx_.size();
    host_to_idx_.try_emplace(tcid_h, host_idx);
    host_to_landmark_block.try_emplace(tcid_h);

    // assumption: every host frame has at least target frame with
    // observations
    // NOTE: in case a host frame loses all of its landmarks due
    // to outlier removal or marginalization of other frames, it becomes quite
    // useless and is expected to be removed before optimization.
    BASALT_ASSERT(!target_map.empty());

    for (const auto& [tcid_t, obs] : target_map) {
      // assumption: every target frame has at least one observation
      BASALT_ASSERT(!obs.empty());

      std::pair<TimeCamId, TimeCamId> key(tcid_h, tcid_t);
      relative_pose_lin.emplace(key, RelPoseLin<Scalar>());
    }
  }

  // Populate lookup for relative poses grouped by host-frame
  for (const auto& [tcid_h, target_map] : lmdb_.getObservations()) {
    // if (used_frames && used_frames->count(tcid_h.frame_id) == 0) continue;
    relative_pose_per_host.emplace_back();

    for (const auto& [tcid_t, _] : target_map) {
      std::pair<TimeCamId, TimeCamId> key(tcid_h, tcid_t);
      auto it = relative_pose_lin.find(key);

      BASALT_ASSERT(it != relative_pose_lin.end());

      relative_pose_per_host.back().emplace_back(it);
    }
  }

  num_cameras = frame_poses.size();

  landmark_ids.clear();
  for (const auto& [k, v] : lmdb_.getLandmarks()) {
    if (used_frames || lost_landmarks) {
      if (used_frames && used_frames->count(v.host_kf_id.frame_id)) {
        landmark_ids.emplace_back(k);
      } else if (lost_landmarks && lost_landmarks->count(k)) {
        landmark_ids.emplace_back(k);
      }
    } else {
      landmark_ids.emplace_back(k);
    }
  }
  size_t num_landmakrs = landmark_ids.size();

  // std::cout << "num_landmakrs " << num_landmakrs << std::endl;

  landmark_blocks.resize(num_landmakrs);

  {
    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        KeypointId lm_id = landmark_ids[r];
        auto& lb = landmark_blocks[r];
        auto& landmark = lmdb_.getLandmark(lm_id);

        lb = LandmarkBlock<Scalar>::template createLandmarkBlock<POSE_SIZE>();

        lb->allocateLandmark(landmark, relative_pose_lin, calib, aom,
                             options.lb_options);
      }
    };

    tbb::blocked_range<size_t> range(0, num_landmakrs);
    tbb::parallel_for(range, body);
  }

  landmark_block_idx.reserve(num_landmakrs);

  num_rows_Q2r = 0;
  for (size_t i = 0; i < num_landmakrs; i++) {
    landmark_block_idx.emplace_back(num_rows_Q2r);
    const auto& lb = landmark_blocks[i];
    num_rows_Q2r += lb->numQ2rows();

    host_to_landmark_block.at(lb->getHostKf()).emplace_back(lb.get());
  }

  if (imu_lin_data) {
    for (const auto& kv : imu_lin_data->imu_meas) {
      imu_blocks.emplace_back(
          new ImuBlock<Scalar>(kv.second, imu_lin_data, aom));
    }
  }

  //    std::cout << "num_rows_Q2r " << num_rows_Q2r << " num_poses " <<
  //    num_cameras
  //              << std::endl;
}

template <typename Scalar, int POSE_SIZE>
LinearizationAbsQR<Scalar, POSE_SIZE>::~LinearizationAbsQR() = default;

template <typename Scalar_, int POSE_SIZE_>
void LinearizationAbsQR<Scalar_, POSE_SIZE_>::log_problem_stats(
    ExecutionStats& stats) const {
  UNUSED(stats);
}

template <typename Scalar, int POSE_SIZE>
Scalar LinearizationAbsQR<Scalar, POSE_SIZE>::linearizeProblem(
    bool* numerically_valid) {
  // reset damping and scaling (might be set from previous iteration)
  pose_damping_diagonal = 0;
  pose_damping_diagonal_sqrt = 0;
  marg_scaling = VecX();

  // Linearize relative poses
  for (const auto& [tcid_h, target_map] : lmdb_.getObservations()) {
    // if (used_frames && used_frames->count(tcid_h.frame_id) == 0) continue;

    for (const auto& [tcid_t, _] : target_map) {
      std::pair<TimeCamId, TimeCamId> key(tcid_h, tcid_t);
      RelPoseLin<Scalar>& rpl = relative_pose_lin.at(key);

      if (tcid_h != tcid_t) {
        const PoseStateWithLin<Scalar>& state_h =
            estimator->getPoseStateWithLin(tcid_h.frame_id);
        const PoseStateWithLin<Scalar>& state_t =
            estimator->getPoseStateWithLin(tcid_t.frame_id);

        // compute relative pose & Jacobians at linearization point
        Sophus::SE3<Scalar> T_t_h_sophus =
            computeRelPose(state_h.getPoseLin(), calib.T_i_c[tcid_h.cam_id],
                           state_t.getPoseLin(), calib.T_i_c[tcid_t.cam_id],
                           &rpl.d_rel_d_h, &rpl.d_rel_d_t);

        // if either state is already linearized, then the current state
        // estimate is different from the linearization point, so recompute
        // the value (not Jacobian) again based on the current state.
        if (state_h.isLinearized() || state_t.isLinearized()) {
          T_t_h_sophus =
              computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                             state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);
        }

        rpl.T_t_h = T_t_h_sophus.matrix();
      } else {
        rpl.T_t_h.setIdentity();
        rpl.d_rel_d_h.setZero();
        rpl.d_rel_d_t.setZero();
      }
    }
  }

  // Linearize landmarks
  size_t num_landmarks = landmark_blocks.size();

  auto body = [&](const tbb::blocked_range<size_t>& range,
                  std::pair<Scalar, bool> error_valid) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      error_valid.first += landmark_blocks[r]->linearizeLandmark();
      error_valid.second =
          error_valid.second && !landmark_blocks[r]->isNumericalFailure();
    }
    return error_valid;
  };

  std::pair<Scalar, bool> initial_value = {0.0, true};
  auto join = [](auto p1, auto p2) {
    p1.first += p2.first;
    p1.second = p1.second && p2.second;
    return p1;
  };

  tbb::blocked_range<size_t> range(0, num_landmarks);
  auto reduction_res = tbb::parallel_reduce(range, initial_value, body, join);

  if (numerically_valid) *numerically_valid = reduction_res.second;

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      reduction_res.first += imu_block->linearizeImu(estimator->frame_states);
    }
  }

  if (marg_lin_data) {
    Scalar marg_prior_error;
    estimator->computeMargPriorError(*marg_lin_data, marg_prior_error);
    reduction_res.first += marg_prior_error;
  }

  return reduction_res.first;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::performQR() {
  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->performQR();
    }
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::setPoseDamping(
    const Scalar lambda) {
  BASALT_ASSERT(lambda >= 0);

  pose_damping_diagonal = lambda;
  pose_damping_diagonal_sqrt = std::sqrt(lambda);
}

template <typename Scalar, int POSE_SIZE>
Scalar LinearizationAbsQR<Scalar, POSE_SIZE>::backSubstitute(
    const VecX& pose_inc) {
  BASALT_ASSERT(pose_inc.size() == signed_cast(aom.total_size));

  auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->backSubstitute(pose_inc, l_diff);
    }
    return l_diff;
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  Scalar l_diff =
      tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      imu_block->backSubstitute(pose_inc, l_diff);
    }
  }

  if (marg_lin_data) {
    size_t marg_size = marg_lin_data->H.cols();
    VecX pose_inc_marg = pose_inc.head(marg_size);

    l_diff += estimator->computeMargPriorModelCostChange(
        *marg_lin_data, marg_scaling, pose_inc_marg);
  }

  return l_diff;
}

template <typename Scalar, int POSE_SIZE>
typename LinearizationAbsQR<Scalar, POSE_SIZE>::VecX
LinearizationAbsQR<Scalar, POSE_SIZE>::getJp_diag2() const {
  // TODO: group relative by host frame

  struct Reductor {
    Reductor(size_t num_rows,
             const std::vector<LandmarkBlockPtr>& landmark_blocks)
        : num_rows_(num_rows), landmark_blocks_(landmark_blocks) {
      res_.setZero(num_rows);
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const auto& lb = landmark_blocks_[r];
        lb->addJp_diag2(res_);
      }
    }

    Reductor(Reductor& a, tbb::split)
        : num_rows_(a.num_rows_), landmark_blocks_(a.landmark_blocks_) {
      res_.setZero(num_rows_);
    };

    inline void join(const Reductor& b) { res_ += b.res_; }

    size_t num_rows_;
    const std::vector<LandmarkBlockPtr>& landmark_blocks_;
    VecX res_;
  };

  Reductor r(aom.total_size, landmark_blocks);

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_reduce(range, r);
  // r(range);

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      imu_block->addJp_diag2(r.res_);
    }
  }

  // TODO: We don't include pose damping here, b/c we use this to compute
  // jacobian scale. Make sure this is clear in the the usage, possibly rename
  // to reflect this, or add assert such that it fails when pose damping is
  // set.

  // Note: ignore damping here

  // Add marginalization prior
  // Asumes marginalization part is in the head. Check for this is located
  // outside
  if (marg_lin_data) {
    size_t marg_size = marg_lin_data->H.cols();
    if (marg_scaling.rows() > 0) {
      r.res_.head(marg_size) += (marg_lin_data->H * marg_scaling.asDiagonal())
                                    .colwise()
                                    .squaredNorm();
    } else {
      r.res_.head(marg_size) += marg_lin_data->H.colwise().squaredNorm();
    }
  }

  return r.res_;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::scaleJl_cols() {
  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->scaleJl_cols();
    }
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::scaleJp_cols(
    const VecX& jacobian_scaling) {
  //    auto body = [&](const tbb::blocked_range<size_t>& range) {
  //      for (size_t r = range.begin(); r != range.end(); ++r) {
  //        landmark_blocks[r]->scaleJp_cols(jacobian_scaling);
  //      }
  //    };

  //    tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  //    tbb::parallel_for(range, body);

  if (true) {
    // In case of absolute poses, we scale Jp in the LMB.

    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks[r]->scaleJp_cols(jacobian_scaling);
      }
    };

    tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
    tbb::parallel_for(range, body);
  } else {
    // In case LMB use relative poses we cannot directly scale the relative pose
    // Jacobians. We have
    //
    //     Jp * diag(S) = Jp_rel * J_rel_to_abs * diag(S)
    //
    // so instead we scale the rel-to-abs jacobians.
    //
    // Note that since we do perform operations like J^T * J on the relative
    // pose Jacobians, we should maybe consider additional scaling like
    //
    //     (Jp_rel * diag(S_rel)) * (diag(S_rel)^-1 * J_rel_to_abs * diag(S)),
    //
    // but this might be only relevant if we do something more sensitive like
    // also include camera intrinsics in the optimization.

    for (auto& [k, v] : relative_pose_lin) {
      size_t h_idx = aom.abs_order_map.at(k.first.frame_id).first;
      size_t t_idx = aom.abs_order_map.at(k.second.frame_id).first;

      v.d_rel_d_h *=
          jacobian_scaling.template segment<POSE_SIZE>(h_idx).asDiagonal();

      v.d_rel_d_t *=
          jacobian_scaling.template segment<POSE_SIZE>(t_idx).asDiagonal();
    }
  }

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      imu_block->scaleJp_cols(jacobian_scaling);
    }
  }

  // Add marginalization scaling
  if (marg_lin_data) {
    // We are only supposed to apply the scaling once.
    BASALT_ASSERT(marg_scaling.size() == 0);

    size_t marg_size = marg_lin_data->H.cols();
    marg_scaling = jacobian_scaling.head(marg_size);
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::setLandmarkDamping(Scalar lambda) {
  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->setLandmarkDamping(lambda);
    }
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r(
    MatX& Q2Jp, VecX& Q2r) const {
  size_t total_size = num_rows_Q2r;
  size_t poses_size = aom.total_size;

  size_t lm_start_idx = 0;

  // Space for IMU data if present
  size_t imu_start_idx = total_size;
  if (imu_lin_data) {
    total_size += imu_lin_data->imu_meas.size() * POSE_VEL_BIAS_SIZE;
  }

  // Space for damping if present
  size_t damping_start_idx = total_size;
  if (hasPoseDamping()) {
    total_size += poses_size;
  }

  // Space for marg-prior if present
  size_t marg_start_idx = total_size;
  if (marg_lin_data) total_size += marg_lin_data->H.rows();

  Q2Jp.setZero(total_size, poses_size);
  Q2r.setZero(total_size);

  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      const auto& lb = landmark_blocks[r];
      lb->get_dense_Q2Jp_Q2r(Q2Jp, Q2r, lm_start_idx + landmark_block_idx[r]);
    }
  };

  // go over all host frames
  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
  // body(range);

  if (imu_lin_data) {
    size_t start_idx = imu_start_idx;
    for (const auto& imu_block : imu_blocks) {
      imu_block->add_dense_Q2Jp_Q2r(Q2Jp, Q2r, start_idx);
      start_idx += POSE_VEL_BIAS_SIZE;
    }
  }

  // Add damping
  get_dense_Q2Jp_Q2r_pose_damping(Q2Jp, damping_start_idx);

  // Add marginalization
  get_dense_Q2Jp_Q2r_marg_prior(Q2Jp, Q2r, marg_start_idx);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_H_b(MatX& H,
                                                          VecX& b) const {
  struct Reductor {
    Reductor(size_t opt_size,
             const std::vector<LandmarkBlockPtr>& landmark_blocks)
        : opt_size_(opt_size), landmark_blocks_(landmark_blocks) {
      H_.setZero(opt_size_, opt_size_);
      b_.setZero(opt_size_);
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        auto& lb = landmark_blocks_[r];
        lb->add_dense_H_b(H_, b_);
      }
    }

    Reductor(Reductor& a, tbb::split)
        : opt_size_(a.opt_size_), landmark_blocks_(a.landmark_blocks_) {
      H_.setZero(opt_size_, opt_size_);
      b_.setZero(opt_size_);
    };

    inline void join(Reductor& b) {
      H_ += b.H_;
      b_ += b.b_;
    }

    size_t opt_size_;
    const std::vector<LandmarkBlockPtr>& landmark_blocks_;

    MatX H_;
    VecX b_;
  };

  size_t opt_size = aom.total_size;

  Reductor r(opt_size, landmark_blocks);

  // go over all host frames
  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_reduce(range, r);

  // Add imu
  add_dense_H_b_imu(r.H_, r.b_);

  // Add damping
  add_dense_H_b_pose_damping(r.H_);

  // Add marginalization
  add_dense_H_b_marg_prior(r.H_, r.b_);

  H = std::move(r.H_);
  b = std::move(r.b_);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r_pose_damping(
    MatX& Q2Jp, size_t start_idx) const {
  size_t poses_size = num_cameras * POSE_SIZE;
  if (hasPoseDamping()) {
    Q2Jp.template block(start_idx, 0, poses_size, poses_size)
        .diagonal()
        .array() = pose_damping_diagonal_sqrt;
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r_marg_prior(
    MatX& Q2Jp, VecX& Q2r, size_t start_idx) const {
  if (!marg_lin_data) return;

  BASALT_ASSERT(marg_lin_data->is_sqrt);

  size_t marg_rows = marg_lin_data->H.rows();
  size_t marg_cols = marg_lin_data->H.cols();

  VecX delta;
  estimator->computeDelta(marg_lin_data->order, delta);

  if (marg_scaling.rows() > 0) {
    Q2Jp.template block(start_idx, 0, marg_rows, marg_cols) =
        marg_lin_data->H * marg_scaling.asDiagonal();
  } else {
    Q2Jp.template block(start_idx, 0, marg_rows, marg_cols) = marg_lin_data->H;
  }

  Q2r.template segment(start_idx, marg_rows) =
      marg_lin_data->H * delta + marg_lin_data->b;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_pose_damping(
    MatX& H) const {
  if (hasPoseDamping()) {
    H.diagonal().array() += pose_damping_diagonal;
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_marg_prior(
    MatX& H, VecX& b) const {
  if (!marg_lin_data) return;

  // Scaling not supported ATM
  BASALT_ASSERT(marg_scaling.rows() == 0);

  Scalar marg_prior_error;
  estimator->linearizeMargPrior(*marg_lin_data, aom, H, b, marg_prior_error);

  //  size_t marg_size = marg_lin_data->H.cols();

  //  VecX delta;
  //  estimator->computeDelta(marg_lin_data->order, delta);

  //  if (marg_scaling.rows() > 0) {
  //    H.topLeftCorner(marg_size, marg_size) +=
  //        marg_scaling.asDiagonal() * marg_lin_data->H.transpose() *
  //        marg_lin_data->H * marg_scaling.asDiagonal();

  //    b.head(marg_size) += marg_scaling.asDiagonal() *
  //                         marg_lin_data->H.transpose() *
  //                         (marg_lin_data->H * delta + marg_lin_data->b);

  //  } else {
  //    H.topLeftCorner(marg_size, marg_size) +=
  //        marg_lin_data->H.transpose() * marg_lin_data->H;

  //    b.head(marg_size) += marg_lin_data->H.transpose() *
  //                         (marg_lin_data->H * delta + marg_lin_data->b);
  //  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_imu(
    DenseAccumulator<Scalar>& accum) const {
  if (!imu_lin_data) return;

  for (const auto& imu_block : imu_blocks) {
    imu_block->add_dense_H_b(accum);
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_imu(MatX& H,
                                                              VecX& b) const {
  if (!imu_lin_data) return;

  // workaround: create an accumulator here, to avoid implementing the
  // add_dense_H_b(H, b) overload in ImuBlock
  DenseAccumulator<Scalar> accum;
  accum.reset(b.size());

  for (const auto& imu_block : imu_blocks) {
    imu_block->add_dense_H_b(accum);
  }

  H += accum.getH();
  b += accum.getB();
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
// Scalar=double, POSE_SIZE=6
template class LinearizationAbsQR<double, 6>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
// Scalar=float, POSE_SIZE=6
template class LinearizationAbsQR<float, 6>;
#endif

}  // namespace basalt
