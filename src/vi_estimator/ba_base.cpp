/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
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

#include <basalt/vi_estimator/ba_base.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <basalt/utils/ba_utils.h>

namespace basalt {

template <class Scalar>
void BundleAdjustmentBase<Scalar>::optimize_single_frame_pose(
    PoseStateWithLin<Scalar>& state_t,
    const std::vector<std::vector<int>>& connected_obs) const {
  const int num_iter = 2;

  struct AbsLinData {
    Mat4 T_t_h;
    Mat6 d_rel_d_t;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  for (int iter = 0; iter < num_iter; iter++) {
    Scalar error = 0;
    Mat6 Ht;
    Vec6 bt;

    Ht.setZero();
    bt.setZero();

    std::unordered_map<std::pair<TimeCamId, TimeCamId>, AbsLinData>
        abs_lin_data;

    for (size_t cam_id = 0; cam_id < connected_obs.size(); cam_id++) {
      TimeCamId tcid_t(state_t.getT_ns(), cam_id);
      for (const auto& lm_id : connected_obs[cam_id]) {
        const Keypoint<Scalar>& kpt_pos = lmdb.getLandmark(lm_id);
        std::pair<TimeCamId, TimeCamId> map_key(kpt_pos.host_kf_id, tcid_t);

        if (abs_lin_data.count(map_key) == 0) {
          const PoseStateWithLin<Scalar>& state_h =
              frame_poses.at(kpt_pos.host_kf_id.frame_id);

          BASALT_ASSERT(kpt_pos.host_kf_id.frame_id != state_t.getT_ns());

          AbsLinData& ald = abs_lin_data[map_key];

          SE3 T_t_h_sophus = computeRelPose<Scalar>(
              state_h.getPose(), calib.T_i_c[kpt_pos.host_kf_id.cam_id],
              state_t.getPose(), calib.T_i_c[cam_id], nullptr, &ald.d_rel_d_t);
          ald.T_t_h = T_t_h_sophus.matrix();
        }
      }
    }

    for (size_t cam_id = 0; cam_id < connected_obs.size(); cam_id++) {
      std::visit(
          [&](const auto& cam) {
            for (const auto& lm_id : connected_obs[cam_id]) {
              TimeCamId tcid_t(state_t.getT_ns(), cam_id);

              const Keypoint<Scalar>& kpt_pos = lmdb.getLandmark(lm_id);
              const Vec2& kpt_obs = kpt_pos.obs.at(tcid_t);
              const AbsLinData& ald =
                  abs_lin_data.at(std::make_pair(kpt_pos.host_kf_id, tcid_t));

              Vec2 res;
              Eigen::Matrix<Scalar, 2, POSE_SIZE> d_res_d_xi;
              bool valid = linearizePoint(kpt_obs, kpt_pos, ald.T_t_h, cam, res,
                                          &d_res_d_xi);

              if (valid) {
                Scalar e = res.norm();
                Scalar huber_weight =
                    e < huber_thresh ? Scalar(1.0) : huber_thresh / e;
                Scalar obs_weight = huber_weight / (obs_std_dev * obs_std_dev);

                error += Scalar(0.5) * (2 - huber_weight) * obs_weight *
                         res.transpose() * res;

                d_res_d_xi *= ald.d_rel_d_t;

                Ht.noalias() += d_res_d_xi.transpose() * d_res_d_xi;
                bt.noalias() += d_res_d_xi.transpose() * res;
              }
            }
          },
          calib.intrinsics[cam_id].variant);
    }

    // Add small damping for GN
    constexpr Scalar lambda = 1e-6;
    Vec6 diag = Ht.diagonal();
    diag *= lambda;
    Ht.diagonal().array() += diag.array().max(lambda);

    // std::cout << "pose opt error " << error << std::endl;
    Vec6 inc = -Ht.ldlt().solve(bt);
    state_t.applyInc(inc);
  }
  // std::cout << "=============================" << std::endl;
}

template <class Scalar_>
void BundleAdjustmentBase<Scalar_>::computeError(
    Scalar& error,
    std::map<int, std::vector<std::pair<TimeCamId, Scalar>>>* outliers,
    Scalar outlier_threshold) const {
  std::vector<TimeCamId> host_frames;
  for (const auto& [tcid, _] : lmdb.getObservations()) {
    host_frames.push_back(tcid);
  }

  tbb::concurrent_unordered_map<int, std::vector<std::pair<TimeCamId, Scalar>>>
      outliers_concurrent;

  auto body = [&](const tbb::blocked_range<size_t>& range, Scalar local_error) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      const TimeCamId& tcid_h = host_frames[r];

      for (const auto& obs_kv : lmdb.getObservations().at(tcid_h)) {
        const TimeCamId& tcid_t = obs_kv.first;

        Mat4 T_t_h;

        if (tcid_h != tcid_t) {
          PoseStateWithLin state_h = getPoseStateWithLin(tcid_h.frame_id);
          PoseStateWithLin state_t = getPoseStateWithLin(tcid_t.frame_id);

          Sophus::SE3<Scalar> T_t_h_sophus =
              computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                             state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);

          T_t_h = T_t_h_sophus.matrix();
        } else {
          T_t_h.setIdentity();
        }

        std::visit(
            [&](const auto& cam) {
              for (KeypointId kpt_id : obs_kv.second) {
                const Keypoint<Scalar>& kpt_pos = lmdb.getLandmark(kpt_id);
                const Vec2& kpt_obs = kpt_pos.obs.at(tcid_t);

                Vec2 res;

                bool valid = linearizePoint(kpt_obs, kpt_pos, T_t_h, cam, res);

                if (valid) {
                  Scalar e = res.norm();

                  if (outliers && e > outlier_threshold) {
                    outliers_concurrent[kpt_id].emplace_back(
                        tcid_t, tcid_h != tcid_t ? e : -2);
                  }

                  Scalar huber_weight =
                      e < huber_thresh ? Scalar(1.0) : huber_thresh / e;
                  Scalar obs_weight =
                      huber_weight / (obs_std_dev * obs_std_dev);

                  local_error += Scalar(0.5) * (2 - huber_weight) * obs_weight *
                                 res.transpose() * res;
                } else {
                  if (outliers) {
                    outliers_concurrent[kpt_id].emplace_back(
                        tcid_t, tcid_h != tcid_t ? -1 : -2);
                  }
                }
              }
            },
            calib.intrinsics[tcid_t.cam_id].variant);
      }
    }

    return local_error;
  };

  tbb::blocked_range<size_t> range(0, host_frames.size());
  Scalar init = 0;
  auto join = std::plus<Scalar>();
  error = tbb::parallel_reduce(range, init, body, join);

  if (outliers) {
    outliers->clear();
    for (auto& [k, v] : outliers_concurrent) {
      outliers->emplace(k, std::move(v));
    }
  }
}

template <class Scalar_>
template <class Scalar2>
void BundleAdjustmentBase<Scalar_>::get_current_points(
    Eigen::aligned_vector<Eigen::Matrix<Scalar2, 3, 1>>& points,
    std::vector<int>& ids) const {
  points.clear();
  ids.clear();

  for (const auto& tcid_host : lmdb.getHostKfs()) {
    Sophus::SE3<Scalar> T_w_i;

    int64_t id = tcid_host.frame_id;
    if (frame_states.count(id) > 0) {
      PoseVelBiasStateWithLin<Scalar> state = frame_states.at(id);
      T_w_i = state.getState().T_w_i;
    } else if (frame_poses.count(id) > 0) {
      PoseStateWithLin<Scalar> state = frame_poses.at(id);

      T_w_i = state.getPose();
    } else {
      std::cout << "Unknown frame id: " << id << std::endl;
      std::abort();
    }

    const Sophus::SE3<Scalar>& T_i_c = calib.T_i_c[tcid_host.cam_id];
    Mat4 T_w_c = (T_w_i * T_i_c).matrix();

    for (const Keypoint<Scalar>* kpt_pos :
         lmdb.getLandmarksForHost(tcid_host)) {
      Vec4 pt_cam = StereographicParam<Scalar>::unproject(kpt_pos->direction);
      pt_cam[3] = kpt_pos->inv_dist;

      Vec4 pt_w = T_w_c * pt_cam;

      points.emplace_back(
          (pt_w.template head<3>() / pt_w[3]).template cast<Scalar2>());
      ids.emplace_back(1);
    }
  }
}

template <class Scalar_>
void BundleAdjustmentBase<Scalar_>::filterOutliers(Scalar outlier_threshold,
                                                   int min_num_obs) {
  Scalar error;
  std::map<int, std::vector<std::pair<TimeCamId, Scalar>>> outliers;
  computeError(error, &outliers, outlier_threshold);

  //  std::cout << "============================================" <<
  //  std::endl; std::cout << "Num landmarks: " << lmdb.numLandmarks() << "
  //  with outliners
  //  "
  //            << outliers.size() << std::endl;

  for (const auto& kv : outliers) {
    int num_obs = lmdb.numObservations(kv.first);
    int num_outliers = kv.second.size();

    bool remove = false;

    if (num_obs - num_outliers < min_num_obs) remove = true;

    //    std::cout << "\tlm_id: " << kv.first << " num_obs: " << num_obs
    //              << " outliers: " << num_outliers << " [";

    for (const auto& kv2 : kv.second) {
      if (kv2.second == -2) remove = true;

      //      std::cout << kv2.second << ", ";
    }

    //    std::cout << "] " << std::endl;

    if (remove) {
      lmdb.removeLandmark(kv.first);
    } else {
      std::set<TimeCamId> outliers;
      for (const auto& kv2 : kv.second) outliers.emplace(kv2.first);
      lmdb.removeObservations(kv.first, outliers);
    }
  }

  // std::cout << "============================================" <<
  // std::endl;
}

template <class Scalar_>
void BundleAdjustmentBase<Scalar_>::computeDelta(const AbsOrderMap& marg_order,
                                                 VecX& delta) const {
  size_t marg_size = marg_order.total_size;
  delta.setZero(marg_size);
  for (const auto& kv : marg_order.abs_order_map) {
    if (kv.second.second == POSE_SIZE) {
      BASALT_ASSERT(frame_poses.at(kv.first).isLinearized());
      delta.template segment<POSE_SIZE>(kv.second.first) =
          frame_poses.at(kv.first).getDelta();
    } else if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      BASALT_ASSERT(frame_states.at(kv.first).isLinearized());
      delta.template segment<POSE_VEL_BIAS_SIZE>(kv.second.first) =
          frame_states.at(kv.first).getDelta();
    } else {
      BASALT_ASSERT(false);
    }
  }
}

template <class Scalar_>
Scalar_ BundleAdjustmentBase<Scalar_>::computeModelCostChange(
    const MatX& H, const VecX& b, const VecX& inc) const {
  // Linearized model cost
  //
  //    L(x) = 0.5 || J*x + r ||^2
  //         = 0.5 x^T J^T J x + x^T J r + 0.5 r^T r
  //         = 0.5 x^T H x + x^T b + 0.5 r^T r,
  //
  // given in normal equation form as
  //
  //    H = J^T J,
  //    b = J^T r.
  //
  // The expected decrease in cost for the computed increment is
  //
  //     l_diff = L(0) - L(inc)
  //            = - 0.5 inc^T H inc - inc^T b
  //            = - inc^T (0.5 H inc + b)

  Scalar l_diff = -inc.dot(Scalar(0.5) * (H * inc) + b);

  return l_diff;
}

template <class Scalar_>
template <class Scalar2>
void BundleAdjustmentBase<Scalar_>::computeProjections(
    std::vector<Eigen::aligned_vector<Eigen::Matrix<Scalar2, 4, 1>>>& data,
    FrameId last_state_t_ns) const {
  for (const auto& kv : lmdb.getObservations()) {
    const TimeCamId& tcid_h = kv.first;

    for (const auto& obs_kv : kv.second) {
      const TimeCamId& tcid_t = obs_kv.first;

      if (tcid_t.frame_id != last_state_t_ns) continue;

      Mat4 T_t_h;
      if (tcid_h != tcid_t) {
        PoseStateWithLin<Scalar> state_h = getPoseStateWithLin(tcid_h.frame_id);
        PoseStateWithLin<Scalar> state_t = getPoseStateWithLin(tcid_t.frame_id);

        Sophus::SE3<Scalar> T_t_h_sophus =
            computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                           state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);

        T_t_h = T_t_h_sophus.matrix();
      } else {
        T_t_h.setIdentity();
      }

      std::visit(
          [&](const auto& cam) {
            for (KeypointId kpt_id : obs_kv.second) {
              const Keypoint<Scalar>& kpt_pos = lmdb.getLandmark(kpt_id);

              Vec2 res;
              Vec4 proj;

              using CamT = std::decay_t<decltype(cam)>;
              linearizePoint<Scalar, CamT>(Vec2::Zero(), kpt_pos, T_t_h, cam,
                                           res, nullptr, nullptr, &proj);

              proj[3] = kpt_id;
              data[tcid_t.cam_id].emplace_back(proj.template cast<Scalar2>());
            }
          },
          calib.intrinsics[tcid_t.cam_id].variant);
    }
  }
}

template <class Scalar_>
void BundleAdjustmentBase<Scalar_>::linearizeMargPrior(
    const MargLinData<Scalar>& mld, const AbsOrderMap& aom, MatX& abs_H,
    VecX& abs_b, Scalar& marg_prior_error) const {
  // Prior is ordered to be in the top left corner of Hessian

  BASALT_ASSERT(size_t(mld.H.cols()) == mld.order.total_size);

  // Check if the order of variables is the same, and it's indeed top-left
  // corner
  for (const auto& kv : mld.order.abs_order_map) {
    UNUSED(aom);
    UNUSED(kv);
    BASALT_ASSERT(aom.abs_order_map.at(kv.first) == kv.second);
    BASALT_ASSERT(kv.second.first < int(mld.order.total_size));
  }

  // Quadratic prior and "delta" of the current state to the original
  // linearization point give cost function
  //
  //    P(x) = 0.5 || J*(delta+x) + r ||^2,
  //
  // alternatively stored in quadratic form as
  //
  //    Hmarg = J^T J,
  //    bmarg = J^T r.
  //
  // This is now to be linearized at x=0, so we get linearization
  //
  //    P(x) = 0.5 || J*x + (J*delta + r) ||^2,
  //
  // with Jacobian J and residual J*delta + r. The normal equations are
  //
  //    H*x + b = 0,
  //    H = J^T J = Hmarg,
  //    b = J^T (J*delta + r) = Hmarg*delta + bmarg.
  //
  // The current cost is
  //
  //    P(0) = 0.5 || J*delta + r ||^2
  //         = 0.5 delta^T J^T J delta + delta^T J^T r + 0.5 r^T r.
  //         = 0.5 delta^T Hmarg delta + delta^T bmarg + 0.5 r^T r.
  //
  // Note: Since the r^T r term does not change with delta, we drop it from the
  // error computation. The main need for the error value is for comparing
  // the cost before and after an update to delta in the optimization loop. This
  // also means the computed error can be negative.

  const size_t marg_size = mld.order.total_size;

  VecX delta;
  computeDelta(mld.order, delta);

  if (mld.is_sqrt) {
    abs_H.topLeftCorner(marg_size, marg_size) += mld.H.transpose() * mld.H;

    abs_b.head(marg_size) += mld.H.transpose() * (mld.b + mld.H * delta);

    marg_prior_error = delta.transpose() * mld.H.transpose() *
                       (Scalar(0.5) * mld.H * delta + mld.b);
  } else {
    abs_H.topLeftCorner(marg_size, marg_size) += mld.H;

    abs_b.head(marg_size) += mld.H * delta + mld.b;

    marg_prior_error =
        delta.transpose() * (Scalar(0.5) * mld.H * delta + mld.b);
  }
}

template <class Scalar_>
void BundleAdjustmentBase<Scalar_>::computeMargPriorError(
    const MargLinData<Scalar>& mld, Scalar& marg_prior_error) const {
  BASALT_ASSERT(size_t(mld.H.cols()) == mld.order.total_size);

  // The current cost is (see above in linearizeMargPrior())
  //
  //    P(0) = 0.5 || J*delta + r ||^2
  //         = 0.5 delta^T J^T J delta + delta^T J^T r + 0.5 r^T r
  //         = 0.5 delta^T Hmarg delta + delta^T bmarg + 0.5 r^T r.
  //
  // Note: Since the r^T r term does not change with delta, we drop it from the
  // error computation. The main need for the error value is for comparing
  // the cost before and after an update to delta in the optimization loop. This
  // also means the computed error can be negative.

  VecX delta;
  computeDelta(mld.order, delta);

  if (mld.is_sqrt) {
    marg_prior_error = delta.transpose() * mld.H.transpose() *
                       (Scalar(0.5) * mld.H * delta + mld.b);
  } else {
    marg_prior_error =
        delta.transpose() * (Scalar(0.5) * mld.H * delta + mld.b);
  }
}

template <class Scalar_>
Scalar_ BundleAdjustmentBase<Scalar_>::computeMargPriorModelCostChange(
    const MargLinData<Scalar>& mld, const VecX& marg_scaling,
    const VecX& marg_pose_inc) const {
  // Quadratic prior and "delta" of the current state to the original
  // linearization point give cost function
  //
  //    P(x) = 0.5 || J*(delta+x) + r ||^2,
  //
  // alternatively stored in quadratic form as
  //
  //    Hmarg = J^T J,
  //    bmarg = J^T r.
  //
  // We want to compute the model cost change. The model function is
  //
  //     L(inc) = P(inc) = 0.5 || J*inc + (J*delta + r) ||^2
  //
  // By setting rlin = J*delta + r we get
  //
  //     L(inc) = 0.5 || J*inc + rlin ||^2
  //            = P(0) + inc^T J^T rlin + 0.5 inc^T J^T J inc
  //
  // and thus the expected decrease in cost for the computed increment is
  //
  //     l_diff = L(0) - L(inc)
  //            = - inc^T J^T rlin - 0.5 inc^T J^T J inc
  //            = - inc^T J^T (rlin + 0.5 J inc)
  //            = - (J inc)^T (rlin + 0.5 (J inc))
  //            = - (J inc)^T (J*delta + r + 0.5 (J inc)).
  //
  // Alternatively, for squared prior storage, we get
  //
  //     l_diff = - inc^T (Hmarg delta + bmarg + 0.5 Hmarg inc)
  //            = - inc^T (Hmarg (delta + 0.5 inc) + bmarg)
  //
  // For Jacobian scaling we need to take extra care. Note that we store the
  // scale separately and don't actually update marg_sqrt_H and marg_sqrt_b
  // in place with the scale. So in the computation above, we need to scale
  // marg_sqrt_H whenever it is multiplied with inc, but NOT when it is
  // multiplied with delta, since delta is also WITHOUT scaling.

  VecX delta;
  computeDelta(mld.order, delta);

  VecX J_inc = marg_pose_inc;
  if (marg_scaling.rows() > 0) J_inc = marg_scaling.asDiagonal() * J_inc;

  Scalar l_diff;

  if (mld.is_sqrt) {
    // No scaling here. This is b part not Jacobian
    const VecX b_Jdelta = mld.H * delta + mld.b;

    J_inc = mld.H * J_inc;
    l_diff = -J_inc.transpose() * (b_Jdelta + Scalar(0.5) * J_inc);
  } else {
    l_diff = -J_inc.dot(mld.H * (delta + Scalar(0.5) * J_inc) + mld.b);
  }

  return l_diff;
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

// Note: double specialization is unconditional, b/c NfrMapper depends on it.
//#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class BundleAdjustmentBase<double>;

template void BundleAdjustmentBase<double>::get_current_points<double>(
    Eigen::aligned_vector<Eigen::Matrix<double, 3, 1>>& points,
    std::vector<int>& ids) const;

template void BundleAdjustmentBase<double>::computeProjections<double>(
    std::vector<Eigen::aligned_vector<Eigen::Matrix<double, 4, 1>>>& data,
    FrameId last_state_t_ns) const;
//#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class BundleAdjustmentBase<float>;

// template void BundleAdjustmentBase<float>::get_current_points<float>(
//    Eigen::aligned_vector<Eigen::Matrix<float, 3, 1>>& points,
//    std::vector<int>& ids) const;

template void BundleAdjustmentBase<float>::get_current_points<double>(
    Eigen::aligned_vector<Eigen::Matrix<double, 3, 1>>& points,
    std::vector<int>& ids) const;

// template void BundleAdjustmentBase<float>::computeProjections<float>(
//    std::vector<Eigen::aligned_vector<Eigen::Matrix<float, 4, 1>>>& data,
//    FrameId last_state_t_ns) const;

template void BundleAdjustmentBase<float>::computeProjections<double>(
    std::vector<Eigen::aligned_vector<Eigen::Matrix<double, 4, 1>>>& data,
    FrameId last_state_t_ns) const;
#endif

}  // namespace basalt
