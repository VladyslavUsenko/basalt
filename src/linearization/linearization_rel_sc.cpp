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

#include <basalt/linearization/linearization_rel_sc.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <basalt/utils/ba_utils.h>
#include <basalt/vi_estimator/sc_ba_base.h>
#include <basalt/linearization/imu_block.hpp>
#include <basalt/utils/cast_utils.hpp>

namespace basalt {

template <typename Scalar, int POSE_SIZE>
LinearizationRelSC<Scalar, POSE_SIZE>::LinearizationRelSC(
    BundleAdjustmentBase<Scalar>* estimator, const AbsOrderMap& aom,
    const Options& options, const MargLinData<Scalar>* marg_lin_data,
    const ImuLinData<Scalar>* imu_lin_data,
    const std::set<FrameId>* used_frames,
    const std::unordered_set<KeypointId>* lost_landmarks,
    int64_t last_state_to_marg)
    : options_(options),
      estimator(estimator),
      lmdb_(estimator->lmdb),
      calib(estimator->calib),
      aom(aom),
      used_frames(used_frames),
      marg_lin_data(marg_lin_data),
      imu_lin_data(imu_lin_data),
      lost_landmarks(lost_landmarks),
      last_state_to_marg(last_state_to_marg),
      pose_damping_diagonal(0),
      pose_damping_diagonal_sqrt(0) {
  BASALT_ASSERT_STREAM(
      options.lb_options.huber_parameter == estimator->huber_thresh,
      "Huber threshold should be set to the same value");

  BASALT_ASSERT_STREAM(options.lb_options.obs_std_dev == estimator->obs_std_dev,
                       "obs_std_dev should be set to the same value");

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
LinearizationRelSC<Scalar, POSE_SIZE>::~LinearizationRelSC() = default;

template <typename Scalar_, int POSE_SIZE_>
void LinearizationRelSC<Scalar_, POSE_SIZE_>::log_problem_stats(
    ExecutionStats& stats) const {
  UNUSED(stats);
}

template <typename Scalar, int POSE_SIZE>
Scalar LinearizationRelSC<Scalar, POSE_SIZE>::linearizeProblem(
    bool* numerically_valid) {
  // reset damping and scaling (might be set from previous iteration)
  pose_damping_diagonal = 0;
  pose_damping_diagonal_sqrt = 0;
  marg_scaling = VecX();

  std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>
      obs_to_lin;

  if (used_frames) {
    const auto& obs = lmdb_.getObservations();

    // select all observations where the host frame is about to be
    // marginalized

    if (lost_landmarks) {
      for (auto it = obs.cbegin(); it != obs.cend(); ++it) {
        if (used_frames->count(it->first.frame_id) > 0) {
          for (auto it2 = it->second.cbegin(); it2 != it->second.cend();
               ++it2) {
            if (it2->first.frame_id <= last_state_to_marg)
              obs_to_lin[it->first].emplace(*it2);
          }
        } else {
          std::map<TimeCamId, std::set<KeypointId>> lost_obs_map;
          for (auto it2 = it->second.cbegin(); it2 != it->second.cend();
               ++it2) {
            for (const auto& lm_id : it2->second) {
              if (lost_landmarks->count(lm_id)) {
                if (it2->first.frame_id <= last_state_to_marg)
                  lost_obs_map[it2->first].emplace(lm_id);
              }
            }
          }
          if (!lost_obs_map.empty()) {
            obs_to_lin[it->first] = lost_obs_map;
          }
        }
      }

    } else {
      for (auto it = obs.cbegin(); it != obs.cend(); ++it) {
        if (used_frames->count(it->first.frame_id) > 0) {
          for (auto it2 = it->second.cbegin(); it2 != it->second.cend();
               ++it2) {
            // TODO: Check how ABS_QR works without last_state_to_marg
            if (it2->first.frame_id <= last_state_to_marg)
              obs_to_lin[it->first].emplace(*it2);
          }
        }
      }
    }
  } else {
    obs_to_lin = lmdb_.getObservations();
  }

  Scalar error;

  ScBundleAdjustmentBase<Scalar>::linearizeHelperStatic(rld_vec, obs_to_lin,
                                                        estimator, error);

  // TODO: Fix the computation of numerically valid points
  if (numerically_valid) *numerically_valid = true;

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      error += imu_block->linearizeImu(estimator->frame_states);
    }
  }

  if (marg_lin_data) {
    Scalar marg_prior_error;
    estimator->computeMargPriorError(*marg_lin_data, marg_prior_error);
    error += marg_prior_error;
  }

  return error;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::performQR() {}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::setPoseDamping(
    const Scalar lambda) {
  BASALT_ASSERT(lambda >= 0);

  pose_damping_diagonal = lambda;
  pose_damping_diagonal_sqrt = std::sqrt(lambda);
}

template <typename Scalar, int POSE_SIZE>
Scalar LinearizationRelSC<Scalar, POSE_SIZE>::backSubstitute(
    const VecX& pose_inc) {
  BASALT_ASSERT(pose_inc.size() == signed_cast(aom.total_size));

  // Update points
  tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
  auto update_points_func = [&](const tbb::blocked_range<size_t>& r,
                                Scalar l_diff) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const auto& rld = rld_vec[i];
      ScBundleAdjustmentBase<Scalar>::updatePoints(aom, rld, -pose_inc, lmdb_,
                                                   &l_diff);
    }

    return l_diff;
  };
  Scalar l_diff = tbb::parallel_reduce(keys_range, Scalar(0),
                                       update_points_func, std::plus<Scalar>());

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
typename LinearizationRelSC<Scalar, POSE_SIZE>::VecX
LinearizationRelSC<Scalar, POSE_SIZE>::getJp_diag2() const {
  // TODO: group relative by host frame

  BASALT_ASSERT_STREAM(false, "Not implemented");
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::scaleJl_cols() {
  BASALT_ASSERT_STREAM(false, "Not implemented");
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::scaleJp_cols(
    const VecX& jacobian_scaling) {
  UNUSED(jacobian_scaling);
  BASALT_ASSERT_STREAM(false, "Not implemented");
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::setLandmarkDamping(Scalar lambda) {
  UNUSED(lambda);
  BASALT_ASSERT_STREAM(false, "Not implemented");
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r(
    MatX& Q2Jp, VecX& Q2r) const {
  MatX H;
  VecX b;
  get_dense_H_b(H, b);

  Eigen::LDLT<Eigen::Ref<MatX>> ldlt(H);

  VecX D_sqrt = ldlt.vectorD().array().max(0).sqrt().matrix();

  // After LDLT, we have
  //     marg_H == P^T*L*D*L^T*P,
  // so we compute the square root as
  //     marg_sqrt_H = sqrt(D)*L^T*P,
  // such that
  //     marg_sqrt_H^T marg_sqrt_H == marg_H.
  Q2Jp.setIdentity(H.rows(), H.cols());
  Q2Jp = ldlt.transpositionsP() * Q2Jp;
  Q2Jp = ldlt.matrixU() * Q2Jp;  // U == L^T
  Q2Jp = D_sqrt.asDiagonal() * Q2Jp;

  // For the right hand side, we want
  //     marg_b == marg_sqrt_H^T * marg_sqrt_b
  // so we compute
  //     marg_sqrt_b = (marg_sqrt_H^T)^-1 * marg_b
  //                 = (P^T*L*sqrt(D))^-1 * marg_b
  //                 = sqrt(D)^-1 * L^-1 * P * marg_b
  Q2r = ldlt.transpositionsP() * b;
  ldlt.matrixL().solveInPlace(Q2r);

  // We already clamped negative values in D_sqrt to 0 above, but for values
  // close to 0 we set b to 0.
  for (int i = 0; i < Q2r.size(); ++i) {
    if (D_sqrt(i) > std::sqrt(std::numeric_limits<Scalar>::min()))
      Q2r(i) /= D_sqrt(i);
    else
      Q2r(i) = 0;
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::get_dense_H_b(MatX& H,
                                                          VecX& b) const {
  typename ScBundleAdjustmentBase<Scalar>::template LinearizeAbsReduce<
      DenseAccumulator<Scalar>>
      lopt_abs(aom);

  tbb::blocked_range<typename Eigen::aligned_vector<RelLinData>::const_iterator>
      range(rld_vec.cbegin(), rld_vec.cend());
  tbb::parallel_reduce(range, lopt_abs);

  // Add imu
  add_dense_H_b_imu(lopt_abs.accum.getH(), lopt_abs.accum.getB());

  // Add damping
  add_dense_H_b_pose_damping(lopt_abs.accum.getH());

  // Add marginalization
  add_dense_H_b_marg_prior(lopt_abs.accum.getH(), lopt_abs.accum.getB());

  H = std::move(lopt_abs.accum.getH());
  b = std::move(lopt_abs.accum.getB());
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r_pose_damping(
    MatX& Q2Jp, size_t start_idx) const {
  UNUSED(Q2Jp);
  UNUSED(start_idx);
  BASALT_ASSERT_STREAM(false, "Not implemented");
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r_marg_prior(
    MatX& Q2Jp, VecX& Q2r, size_t start_idx) const {
  if (!marg_lin_data) return;

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
void LinearizationRelSC<Scalar, POSE_SIZE>::add_dense_H_b_pose_damping(
    MatX& H) const {
  if (hasPoseDamping()) {
    H.diagonal().array() += pose_damping_diagonal;
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::add_dense_H_b_marg_prior(
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
void LinearizationRelSC<Scalar, POSE_SIZE>::add_dense_H_b_imu(
    DenseAccumulator<Scalar>& accum) const {
  if (!imu_lin_data) return;

  for (const auto& imu_block : imu_blocks) {
    imu_block->add_dense_H_b(accum);
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationRelSC<Scalar, POSE_SIZE>::add_dense_H_b_imu(MatX& H,
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
template class LinearizationRelSC<double, 6>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
// Scalar=float, POSE_SIZE=6
template class LinearizationRelSC<float, 6>;
#endif

}  // namespace basalt
