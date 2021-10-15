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

#include <basalt/vi_estimator/sc_ba_base.h>

#include <tbb/parallel_for.h>

#include <basalt/imu/preintegration.h>
#include <basalt/utils/ba_utils.h>

namespace basalt {

template <class Scalar_>
void ScBundleAdjustmentBase<Scalar_>::updatePoints(
    const AbsOrderMap& aom, const RelLinData& rld, const VecX& inc,
    LandmarkDatabase<Scalar>& lmdb, Scalar* l_diff) {
  // We want to compute the model cost change. The model fuction is
  //
  //    L(inc) = 0.5 r^T r + inc^T b + 0.5 inc^T H inc,
  //
  // and thus the expected decrease in cost for the computed increment is
  //
  //    l_diff = L(0) - L(inc)
  //           = - inc^T b - 0.5 inc^T H inc.
  //
  // Here we have
  //
  //          | incp |       | bp |       | Hpp  Hpl |
  //    inc = |      |   b = |    |   H = |          |
  //          | incl |,      | bl |,      | Hlp  Hll |,
  //
  // and thus
  //
  //    l_diff = - incp^T bp - incl^T bl -
  //               0.5 incp^T Hpp incp - incp^T Hpl incl - 0.5 incl^T Hll incl
  //           = - sum_{p} (incp^T (bp + 0.5 Hpp incp)) -
  //               sum_{l} (incl^T (bl + 0.5 Hll incl +
  //                                sum_{obs} (Hpl^T incp))),
  //
  // where sum_{p} sums over all (relative) poses, sum_{l} over all landmarks,
  // and sum_{obs} over all observations of each landmark.
  //
  // Note: l_diff acts like an accumulator; we just add w/o initializing to 0.

  VecX rel_inc;
  rel_inc.setZero(rld.order.size() * POSE_SIZE);
  for (size_t i = 0; i < rld.order.size(); i++) {
    const TimeCamId& tcid_h = rld.order[i].first;
    const TimeCamId& tcid_t = rld.order[i].second;

    if (tcid_h.frame_id != tcid_t.frame_id) {
      int abs_h_idx = aom.abs_order_map.at(tcid_h.frame_id).first;
      int abs_t_idx = aom.abs_order_map.at(tcid_t.frame_id).first;

      Eigen::Matrix<Scalar, POSE_SIZE, 1> inc_p =
          rld.d_rel_d_h[i] * inc.template segment<POSE_SIZE>(abs_h_idx) +
          rld.d_rel_d_t[i] * inc.template segment<POSE_SIZE>(abs_t_idx);

      rel_inc.template segment<POSE_SIZE>(i * POSE_SIZE) = inc_p;

      if (l_diff) {
        // Note: inc_p is still negated b/c we solve H(-x) = b
        const FrameRelLinData& frld = rld.Hpppl[i];
        *l_diff -= -inc_p.dot((frld.bp - Scalar(0.5) * (frld.Hpp * inc_p)));
      }
    }
  }

  for (const auto& kv : rld.lm_to_obs) {
    int lm_idx = kv.first;
    const auto& lm_obs = kv.second;

    Vec3 H_l_p_x;
    H_l_p_x.setZero();

    for (size_t k = 0; k < lm_obs.size(); k++) {
      int rel_idx = lm_obs[k].first;
      const FrameRelLinData& frld = rld.Hpppl.at(rel_idx);
      const Eigen::Matrix<Scalar, POSE_SIZE, 3>& H_p_l_other =
          frld.Hpl.at(lm_obs[k].second);

      // Note: pose increment is still negated b/c we solve "H (-inc) = b"
      H_l_p_x += H_p_l_other.transpose() *
                 rel_inc.template segment<POSE_SIZE>(rel_idx * POSE_SIZE);
    }

    // final negation of inc_l b/c we solve "H (-inc) = b"
    Vec3 inc_l = -(rld.Hllinv.at(lm_idx) * (rld.bl.at(lm_idx) - H_l_p_x));

    Keypoint<Scalar>& kpt = lmdb.getLandmark(lm_idx);
    kpt.direction += inc_l.template head<2>();
    kpt.inv_dist += inc_l[2];

    kpt.inv_dist = std::max(Scalar(0), kpt.inv_dist);

    if (l_diff) {
      // Note: rel_inc and thus H_l_p_x are still negated b/c we solve H(-x) = b
      *l_diff -= inc_l.transpose() *
                 (rld.bl.at(lm_idx) +
                  Scalar(0.5) * (rld.Hll.at(lm_idx) * inc_l) - H_l_p_x);
    }
  }
}

template <class Scalar_>
void ScBundleAdjustmentBase<Scalar_>::updatePointsAbs(
    const AbsOrderMap& aom, const AbsLinData& ald, const VecX& inc,
    LandmarkDatabase<Scalar>& lmdb, Scalar* l_diff) {
  // We want to compute the model cost change. The model fuction is
  //
  //    L(inc) = 0.5 r^T r + inc^T b + 0.5 inc^T H inc,
  //
  // and thus the expected decrease in cost for the computed increment is
  //
  //    l_diff = L(0) - L(inc)
  //           = - inc^T b - 0.5 inc^T H inc.
  //
  // Here we have
  //
  //          | incp |       | bp |       | Hpp  Hpl |
  //    inc = |      |   b = |    |   H = |          |
  //          | incl |,      | bl |,      | Hlp  Hll |,
  //
  // and thus
  //
  //    l_diff = - incp^T bp - incl^T bl -
  //               0.5 incp^T Hpp incp - incp^T Hpl incl - 0.5 incl^T Hll incl
  //           = - sum_{p} (incp^T (bp + 0.5 Hpp incp)) -
  //               sum_{l} (incl^T (bl + 0.5 Hll incl +
  //                                sum_{obs} (Hpl^T incp))),
  //
  // where sum_{p} sums over all (relative) poses, sum_{l} over all landmarks,
  // and sum_{obs} over all observations of each landmark.
  //
  // Note: l_diff acts like an accumulator; we just add w/o initializing to 0.

  if (l_diff) {
    for (size_t i = 0; i < ald.order.size(); i++) {
      const TimeCamId& tcid_h = ald.order[i].first;
      const TimeCamId& tcid_t = ald.order[i].second;

      int abs_h_idx = aom.abs_order_map.at(tcid_h.frame_id).first;
      int abs_t_idx = aom.abs_order_map.at(tcid_t.frame_id).first;

      auto inc_h = inc.template segment<POSE_SIZE>(abs_h_idx);
      auto inc_t = inc.template segment<POSE_SIZE>(abs_t_idx);

      // Note: inc_p is still negated b/c we solve H(-x) = b
      const FrameAbsLinData& fald = ald.Hpppl[i];
      *l_diff -= -inc_h.dot((fald.bph - Scalar(0.5) * (fald.Hphph * inc_h)) -
                            fald.Hphpt * inc_t) -
                 inc_t.dot((fald.bpt - Scalar(0.5) * (fald.Hptpt * inc_t)));
    }
  }

  for (const auto& kv : ald.lm_to_obs) {
    int lm_idx = kv.first;
    const auto& lm_obs = kv.second;

    Vec3 H_l_p_x;
    H_l_p_x.setZero();

    for (size_t k = 0; k < lm_obs.size(); k++) {
      int rel_idx = lm_obs[k].first;
      const TimeCamId& tcid_h = ald.order.at(rel_idx).first;
      const TimeCamId& tcid_t = ald.order.at(rel_idx).second;

      int abs_h_idx = aom.abs_order_map.at(tcid_h.frame_id).first;
      int abs_t_idx = aom.abs_order_map.at(tcid_t.frame_id).first;

      auto inc_h = inc.template segment<POSE_SIZE>(abs_h_idx);
      auto inc_t = inc.template segment<POSE_SIZE>(abs_t_idx);

      const FrameAbsLinData& fald = ald.Hpppl.at(rel_idx);
      const Eigen::Matrix<Scalar, POSE_SIZE, 3>& H_ph_l_other =
          fald.Hphl.at(lm_obs[k].second);
      const Eigen::Matrix<Scalar, POSE_SIZE, 3>& H_pt_l_other =
          fald.Hptl.at(lm_obs[k].second);

      // Note: pose increment is still negated b/c we solve "H (-inc) = b"
      H_l_p_x += H_ph_l_other.transpose() * inc_h;
      H_l_p_x += H_pt_l_other.transpose() * inc_t;
    }

    // final negation of inc_l b/c we solve "H (-inc) = b"
    Vec3 inc_l = -(ald.Hllinv.at(lm_idx) * (ald.bl.at(lm_idx) - H_l_p_x));

    Keypoint<Scalar>& kpt = lmdb.getLandmark(lm_idx);
    kpt.direction += inc_l.template head<2>();
    kpt.inv_dist += inc_l[2];

    kpt.inv_dist = std::max(Scalar(0), kpt.inv_dist);

    if (l_diff) {
      // Note: rel_inc and thus H_l_p_x are still negated b/c we solve H(-x) = b
      *l_diff -= inc_l.transpose() *
                 (ald.bl.at(lm_idx) +
                  Scalar(0.5) * (ald.Hll.at(lm_idx) * inc_l) - H_l_p_x);
    }
  }
}

template <class Scalar_>
void ScBundleAdjustmentBase<Scalar_>::linearizeHelperStatic(
    Eigen::aligned_vector<RelLinData>& rld_vec,
    const std::unordered_map<
        TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
    const BundleAdjustmentBase<Scalar>* ba_base, Scalar& error) {
  error = 0;

  rld_vec.clear();

  std::vector<TimeCamId> obs_tcid_vec;
  for (const auto& kv : obs_to_lin) {
    obs_tcid_vec.emplace_back(kv.first);
    rld_vec.emplace_back(ba_base->lmdb.numLandmarks(), kv.second.size());
  }

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, obs_tcid_vec.size()),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          auto kv = obs_to_lin.find(obs_tcid_vec[r]);

          RelLinData& rld = rld_vec[r];

          rld.error = Scalar(0);

          const TimeCamId& tcid_h = kv->first;

          for (const auto& obs_kv : kv->second) {
            const TimeCamId& tcid_t = obs_kv.first;
            rld.order.emplace_back(std::make_pair(tcid_h, tcid_t));

            Mat4 T_t_h;

            if (tcid_h != tcid_t) {
              // target and host are not the same
              PoseStateWithLin state_h =
                  ba_base->getPoseStateWithLin(tcid_h.frame_id);
              PoseStateWithLin state_t =
                  ba_base->getPoseStateWithLin(tcid_t.frame_id);

              Mat6 d_rel_d_h, d_rel_d_t;

              SE3 T_t_h_sophus = computeRelPose(
                  state_h.getPoseLin(), ba_base->calib.T_i_c[tcid_h.cam_id],
                  state_t.getPoseLin(), ba_base->calib.T_i_c[tcid_t.cam_id],
                  &d_rel_d_h, &d_rel_d_t);

              rld.d_rel_d_h.emplace_back(d_rel_d_h);
              rld.d_rel_d_t.emplace_back(d_rel_d_t);

              if (state_h.isLinearized() || state_t.isLinearized()) {
                T_t_h_sophus = computeRelPose(
                    state_h.getPose(), ba_base->calib.T_i_c[tcid_h.cam_id],
                    state_t.getPose(), ba_base->calib.T_i_c[tcid_t.cam_id]);
              }

              T_t_h = T_t_h_sophus.matrix();
            } else {
              T_t_h.setIdentity();
              rld.d_rel_d_h.emplace_back(Mat6::Zero());
              rld.d_rel_d_t.emplace_back(Mat6::Zero());
            }

            FrameRelLinData frld;

            std::visit(
                [&](const auto& cam) {
                  for (KeypointId kpt_id : obs_kv.second) {
                    const Keypoint<Scalar>& kpt_pos =
                        ba_base->lmdb.getLandmark(kpt_id);
                    const Vec2& kpt_obs = kpt_pos.obs.at(tcid_t);

                    Vec2 res;
                    Eigen::Matrix<Scalar, 2, POSE_SIZE> d_res_d_xi;
                    Eigen::Matrix<Scalar, 2, 3> d_res_d_p;

                    bool valid = linearizePoint(kpt_obs, kpt_pos, T_t_h, cam,
                                                res, &d_res_d_xi, &d_res_d_p);

                    if (valid) {
                      Scalar e = res.norm();
                      Scalar huber_weight = e < ba_base->huber_thresh
                                                ? Scalar(1.0)
                                                : ba_base->huber_thresh / e;
                      Scalar obs_weight = huber_weight / (ba_base->obs_std_dev *
                                                          ba_base->obs_std_dev);

                      rld.error += Scalar(0.5) * (2 - huber_weight) *
                                   obs_weight * res.transpose() * res;

                      if (rld.Hll.count(kpt_id) == 0) {
                        rld.Hll[kpt_id].setZero();
                        rld.bl[kpt_id].setZero();
                      }

                      rld.Hll[kpt_id] +=
                          obs_weight * d_res_d_p.transpose() * d_res_d_p;
                      rld.bl[kpt_id] +=
                          obs_weight * d_res_d_p.transpose() * res;

                      frld.Hpp +=
                          obs_weight * d_res_d_xi.transpose() * d_res_d_xi;
                      frld.bp += obs_weight * d_res_d_xi.transpose() * res;
                      frld.Hpl.emplace_back(obs_weight *
                                            d_res_d_xi.transpose() * d_res_d_p);
                      frld.lm_id.emplace_back(kpt_id);

                      rld.lm_to_obs[kpt_id].emplace_back(rld.Hpppl.size(),
                                                         frld.lm_id.size() - 1);
                    }
                  }
                },
                ba_base->calib.intrinsics[tcid_t.cam_id].variant);

            rld.Hpppl.emplace_back(frld);
          }

          rld.invert_keypoint_hessians();
        }
      });

  for (const auto& rld : rld_vec) error += rld.error;
}

template <class Scalar_>
void ScBundleAdjustmentBase<Scalar_>::linearizeHelperAbsStatic(
    Eigen::aligned_vector<AbsLinData>& ald_vec,
    const std::unordered_map<
        TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
    const BundleAdjustmentBase<Scalar>* ba_base, Scalar& error) {
  error = 0;

  ald_vec.clear();

  std::vector<TimeCamId> obs_tcid_vec;
  for (const auto& kv : obs_to_lin) {
    obs_tcid_vec.emplace_back(kv.first);
    ald_vec.emplace_back(ba_base->lmdb.numLandmarks(), kv.second.size());
  }

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, obs_tcid_vec.size()),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          auto kv = obs_to_lin.find(obs_tcid_vec[r]);

          AbsLinData& ald = ald_vec[r];

          ald.error = Scalar(0);

          const TimeCamId& tcid_h = kv->first;

          for (const auto& obs_kv : kv->second) {
            const TimeCamId& tcid_t = obs_kv.first;
            ald.order.emplace_back(std::make_pair(tcid_h, tcid_t));

            Mat4 T_t_h;
            Mat6 d_rel_d_h, d_rel_d_t;

            if (tcid_h != tcid_t) {
              // target and host are not the same
              PoseStateWithLin state_h =
                  ba_base->getPoseStateWithLin(tcid_h.frame_id);
              PoseStateWithLin state_t =
                  ba_base->getPoseStateWithLin(tcid_t.frame_id);

              SE3 T_t_h_sophus = computeRelPose(
                  state_h.getPoseLin(), ba_base->calib.T_i_c[tcid_h.cam_id],
                  state_t.getPoseLin(), ba_base->calib.T_i_c[tcid_t.cam_id],
                  &d_rel_d_h, &d_rel_d_t);

              if (state_h.isLinearized() || state_t.isLinearized()) {
                T_t_h_sophus = computeRelPose(
                    state_h.getPose(), ba_base->calib.T_i_c[tcid_h.cam_id],
                    state_t.getPose(), ba_base->calib.T_i_c[tcid_t.cam_id]);
              }

              T_t_h = T_t_h_sophus.matrix();
            } else {
              T_t_h.setIdentity();
              d_rel_d_h.setZero();
              d_rel_d_t.setZero();
            }

            FrameAbsLinData fald;

            std::visit(
                [&](const auto& cam) {
                  for (KeypointId kpt_id : obs_kv.second) {
                    const Keypoint<Scalar>& kpt_pos =
                        ba_base->lmdb.getLandmark(kpt_id);
                    const Vec2& kpt_obs = kpt_pos.obs.at(tcid_t);

                    Vec2 res;
                    Eigen::Matrix<Scalar, 2, POSE_SIZE> d_res_d_xi_h,
                        d_res_d_xi_t;
                    Eigen::Matrix<Scalar, 2, 3> d_res_d_p;

                    bool valid;
                    {
                      Eigen::Matrix<Scalar, 2, POSE_SIZE> d_res_d_xi;

                      valid = linearizePoint(kpt_obs, kpt_pos, T_t_h, cam, res,
                                             &d_res_d_xi, &d_res_d_p);

                      d_res_d_xi_h = d_res_d_xi * d_rel_d_h;
                      d_res_d_xi_t = d_res_d_xi * d_rel_d_t;
                    }

                    if (valid) {
                      Scalar e = res.norm();
                      Scalar huber_weight = e < ba_base->huber_thresh
                                                ? Scalar(1.0)
                                                : ba_base->huber_thresh / e;
                      Scalar obs_weight = huber_weight / (ba_base->obs_std_dev *
                                                          ba_base->obs_std_dev);

                      ald.error += Scalar(0.5) * (2 - huber_weight) *
                                   obs_weight * res.transpose() * res;

                      if (ald.Hll.count(kpt_id) == 0) {
                        ald.Hll[kpt_id].setZero();
                        ald.bl[kpt_id].setZero();
                      }

                      ald.Hll[kpt_id] +=
                          obs_weight * d_res_d_p.transpose() * d_res_d_p;
                      ald.bl[kpt_id] +=
                          obs_weight * d_res_d_p.transpose() * res;

                      fald.Hphph +=
                          obs_weight * d_res_d_xi_h.transpose() * d_res_d_xi_h;
                      fald.Hptpt +=
                          obs_weight * d_res_d_xi_t.transpose() * d_res_d_xi_t;
                      fald.Hphpt +=
                          obs_weight * d_res_d_xi_h.transpose() * d_res_d_xi_t;

                      fald.bph += obs_weight * d_res_d_xi_h.transpose() * res;
                      fald.bpt += obs_weight * d_res_d_xi_t.transpose() * res;

                      fald.Hphl.emplace_back(
                          obs_weight * d_res_d_xi_h.transpose() * d_res_d_p);
                      fald.Hptl.emplace_back(
                          obs_weight * d_res_d_xi_t.transpose() * d_res_d_p);

                      fald.lm_id.emplace_back(kpt_id);

                      ald.lm_to_obs[kpt_id].emplace_back(ald.Hpppl.size(),
                                                         fald.lm_id.size() - 1);
                    }
                  }
                },
                ba_base->calib.intrinsics[tcid_t.cam_id].variant);

            ald.Hpppl.emplace_back(fald);
          }

          ald.invert_keypoint_hessians();
        }
      });

  for (const auto& rld : ald_vec) error += rld.error;
}

template <class Scalar_>
void ScBundleAdjustmentBase<Scalar_>::linearizeRel(const RelLinData& rld,
                                                   MatX& H, VecX& b) {
  //  std::cout << "linearizeRel: KF " << frame_states.size() << " obs "
  //            << obs.size() << std::endl;

  // Do schur complement
  size_t msize = rld.order.size();
  H.setZero(POSE_SIZE * msize, POSE_SIZE * msize);
  b.setZero(POSE_SIZE * msize);

  for (size_t i = 0; i < rld.order.size(); i++) {
    const FrameRelLinData& frld = rld.Hpppl.at(i);

    H.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i, POSE_SIZE * i) +=
        frld.Hpp;
    b.template segment<POSE_SIZE>(POSE_SIZE * i) += frld.bp;

    for (size_t j = 0; j < frld.lm_id.size(); j++) {
      Eigen::Matrix<Scalar, POSE_SIZE, 3> H_pl_H_ll_inv;
      int lm_id = frld.lm_id[j];

      H_pl_H_ll_inv = frld.Hpl[j] * rld.Hllinv.at(lm_id);
      b.template segment<POSE_SIZE>(POSE_SIZE * i) -=
          H_pl_H_ll_inv * rld.bl.at(lm_id);

      const auto& other_obs = rld.lm_to_obs.at(lm_id);
      for (size_t k = 0; k < other_obs.size(); k++) {
        const FrameRelLinData& frld_other = rld.Hpppl.at(other_obs[k].first);
        int other_i = other_obs[k].first;

        Eigen::Matrix<Scalar, 3, POSE_SIZE> H_l_p_other =
            frld_other.Hpl[other_obs[k].second].transpose();

        H.template block<POSE_SIZE, POSE_SIZE>(
            POSE_SIZE * i, POSE_SIZE * other_i) -= H_pl_H_ll_inv * H_l_p_other;
      }
    }
  }
}

template <class Scalar_>
Eigen::VectorXd ScBundleAdjustmentBase<Scalar_>::checkNullspace(
    const MatX& H, const VecX& b, const AbsOrderMap& order,
    const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
        frame_states,
    const Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>>& frame_poses,
    bool verbose) {
  using Vec3d = Eigen::Vector3d;
  using VecXd = Eigen::VectorXd;
  using Mat3d = Eigen::Matrix3d;
  using MatXd = Eigen::MatrixXd;
  using SO3d = Sophus::SO3d;

  BASALT_ASSERT(size_t(H.cols()) == order.total_size);
  size_t marg_size = order.total_size;

  // Idea: We construct increments that we know should lie in the null-space of
  // the prior from marginalized observations (except for the initial pose prior
  // we set at initialization) --> shift global translations (x,y,z separately),
  // or global rotations (r,p,y separately); for VIO only yaw rotation shift is
  // in nullspace. Compared to a random increment, we expect them to stay small
  // (at initial level of the initial pose prior). If they increase over time,
  // we accumulate spurious information on unobservable degrees of freedom.
  //
  // Poses are cam-to-world and we use left-increment in optimization, so
  // perturbations are in world frame. Hence translational increments are
  // independent. For rotational increments we also need to rotate translations
  // and velocities, both of which are expressed in world frame. For
  // translations, we move the center of rotation to the camera center centroid
  // for better numerics.

  VecXd inc_x, inc_y, inc_z, inc_roll, inc_pitch, inc_yaw;
  inc_x.setZero(marg_size);
  inc_y.setZero(marg_size);
  inc_z.setZero(marg_size);
  inc_roll.setZero(marg_size);
  inc_pitch.setZero(marg_size);
  inc_yaw.setZero(marg_size);

  int num_trans = 0;
  Vec3d mean_trans;
  mean_trans.setZero();

  // Compute mean translation
  for (const auto& kv : order.abs_order_map) {
    Vec3d trans;
    if (kv.second.second == POSE_SIZE) {
      mean_trans += frame_poses.at(kv.first)
                        .getPoseLin()
                        .translation()
                        .template cast<double>();
      num_trans++;
    } else if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      mean_trans += frame_states.at(kv.first)
                        .getStateLin()
                        .T_w_i.translation()
                        .template cast<double>();
      num_trans++;
    } else {
      std::cerr << "Unknown size of the state: " << kv.second.second
                << std::endl;
      std::abort();
    }
  }
  mean_trans /= num_trans;

  double eps = 0.01;

  // Compute nullspace increments
  for (const auto& kv : order.abs_order_map) {
    inc_x(kv.second.first + 0) = eps;
    inc_y(kv.second.first + 1) = eps;
    inc_z(kv.second.first + 2) = eps;
    inc_roll(kv.second.first + 3) = eps;
    inc_pitch(kv.second.first + 4) = eps;
    inc_yaw(kv.second.first + 5) = eps;

    Vec3d trans;
    if (kv.second.second == POSE_SIZE) {
      trans = frame_poses.at(kv.first)
                  .getPoseLin()
                  .translation()
                  .template cast<double>();
    } else if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      trans = frame_states.at(kv.first)
                  .getStateLin()
                  .T_w_i.translation()
                  .template cast<double>();
    } else {
      BASALT_ASSERT(false);
    }

    // combine rotation with global translation to make it rotation around
    // translation centroid, not around origin (better numerics). Note that
    // velocities are not affected by global translation.
    trans -= mean_trans;

    // Jacobian of translation w.r.t. the rotation increment (one column each
    // for the 3 different increments)
    Mat3d J = -SO3d::hat(trans);
    J *= eps;

    inc_roll.template segment<3>(kv.second.first) = J.col(0);
    inc_pitch.template segment<3>(kv.second.first) = J.col(1);
    inc_yaw.template segment<3>(kv.second.first) = J.col(2);

    if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      Vec3d vel = frame_states.at(kv.first)
                      .getStateLin()
                      .vel_w_i.template cast<double>();

      // Jacobian of velocity w.r.t. the rotation increment (one column each
      // for the 3 different increments)
      Mat3d J_vel = -SO3d::hat(vel);
      J_vel *= eps;

      inc_roll.template segment<3>(kv.second.first + POSE_SIZE) = J_vel.col(0);
      inc_pitch.template segment<3>(kv.second.first + POSE_SIZE) = J_vel.col(1);
      inc_yaw.template segment<3>(kv.second.first + POSE_SIZE) = J_vel.col(2);
    }
  }

  inc_x.normalize();
  inc_y.normalize();
  inc_z.normalize();
  inc_roll.normalize();
  inc_pitch.normalize();
  inc_yaw.normalize();

  //  std::cout << "inc_x   " << inc_x.transpose() << std::endl;
  //  std::cout << "inc_y   " << inc_y.transpose() << std::endl;
  //  std::cout << "inc_z   " << inc_z.transpose() << std::endl;
  //  std::cout << "inc_yaw " << inc_yaw.transpose() << std::endl;

  VecXd inc_random;
  inc_random.setRandom(marg_size);
  inc_random.normalize();

  MatXd H_d = H.template cast<double>();
  VecXd b_d = b.template cast<double>();

  VecXd xHx(7);
  VecXd xb(7);

  xHx[0] = inc_x.transpose() * H_d * inc_x;
  xHx[1] = inc_y.transpose() * H_d * inc_y;
  xHx[2] = inc_z.transpose() * H_d * inc_z;
  xHx[3] = inc_roll.transpose() * H_d * inc_roll;
  xHx[4] = inc_pitch.transpose() * H_d * inc_pitch;
  xHx[5] = inc_yaw.transpose() * H_d * inc_yaw;
  xHx[6] = inc_random.transpose() * H_d * inc_random;

  // b == J^T r, so the increments should also lie in left-nullspace of b
  xb[0] = inc_x.transpose() * b_d;
  xb[1] = inc_y.transpose() * b_d;
  xb[2] = inc_z.transpose() * b_d;
  xb[3] = inc_roll.transpose() * b_d;
  xb[4] = inc_pitch.transpose() * b_d;
  xb[5] = inc_yaw.transpose() * b_d;
  xb[6] = inc_random.transpose() * b_d;

  if (verbose) {
    std::cout << "nullspace x_trans: " << xHx[0] << " + " << xb[0] << std::endl;
    std::cout << "nullspace y_trans: " << xHx[1] << " + " << xb[1] << std::endl;
    std::cout << "nullspace z_trans: " << xHx[2] << " + " << xb[2] << std::endl;
    std::cout << "nullspace roll   : " << xHx[3] << " + " << xb[3] << std::endl;
    std::cout << "nullspace pitch  : " << xHx[4] << " + " << xb[4] << std::endl;
    std::cout << "nullspace yaw    : " << xHx[5] << " + " << xb[5] << std::endl;
    std::cout << "nullspace random : " << xHx[6] << " + " << xb[6] << std::endl;
  }

  return xHx + xb;
}

template <class Scalar_>
Eigen::VectorXd ScBundleAdjustmentBase<Scalar_>::checkEigenvalues(
    const MatX& H, bool verbose) {
  // For EV, we use SelfAdjointEigenSolver to avoid getting (numerically)
  // complex eigenvalues.

  // We do this computation in double precision to avoid any inaccuracies that
  // come from the squaring.

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      H.template cast<double>());
  if (eigensolver.info() != Eigen::Success) {
    BASALT_LOG_FATAL("eigen solver failed");
  }

  if (verbose) {
    std::cout << "EV:\n" << eigensolver.eigenvalues() << std::endl;
  }

  return eigensolver.eigenvalues();
}

template <class Scalar_>
void ScBundleAdjustmentBase<Scalar_>::computeImuError(
    const AbsOrderMap& aom, Scalar& imu_error, Scalar& bg_error,
    Scalar& ba_error,
    const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>& states,
    const Eigen::aligned_map<int64_t, IntegratedImuMeasurement<Scalar>>&
        imu_meas,
    const Vec3& gyro_bias_weight, const Vec3& accel_bias_weight,
    const Vec3& g) {
  imu_error = 0;
  bg_error = 0;
  ba_error = 0;
  for (const auto& kv : imu_meas) {
    if (kv.second.get_dt_ns() != 0) {
      int64_t start_t = kv.second.get_start_t_ns();
      int64_t end_t = kv.second.get_start_t_ns() + kv.second.get_dt_ns();

      if (aom.abs_order_map.count(start_t) == 0 ||
          aom.abs_order_map.count(end_t) == 0)
        continue;

      PoseVelBiasStateWithLin<Scalar> start_state = states.at(start_t);
      PoseVelBiasStateWithLin<Scalar> end_state = states.at(end_t);

      const typename PoseVelState<Scalar>::VecN res = kv.second.residual(
          start_state.getState(), g, end_state.getState(),
          start_state.getState().bias_gyro, start_state.getState().bias_accel);

      //      std::cout << "res: (" << start_t << "," << end_t << ") "
      //                << res.transpose() << std::endl;

      //      std::cerr << "cov_inv:\n" << kv.second.get_cov_inv() <<
      //      std::endl;

      imu_error +=
          Scalar(0.5) * res.transpose() * kv.second.get_cov_inv() * res;

      Scalar dt = kv.second.get_dt_ns() * Scalar(1e-9);
      {
        Vec3 gyro_bias_weight_dt = gyro_bias_weight / dt;
        Vec3 res_bg =
            start_state.getState().bias_gyro - end_state.getState().bias_gyro;

        bg_error += Scalar(0.5) * res_bg.transpose() *
                    gyro_bias_weight_dt.asDiagonal() * res_bg;
      }

      {
        Vec3 accel_bias_weight_dt = accel_bias_weight / dt;
        Vec3 res_ba =
            start_state.getState().bias_accel - end_state.getState().bias_accel;

        ba_error += Scalar(0.5) * res_ba.transpose() *
                    accel_bias_weight_dt.asDiagonal() * res_ba;
      }
    }
  }
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

// Note: double specialization is unconditional, b/c NfrMapper depends on it.
//#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class ScBundleAdjustmentBase<double>;
//#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class ScBundleAdjustmentBase<float>;
#endif

}  // namespace basalt
