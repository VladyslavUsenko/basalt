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
#pragma once

#include <basalt/vi_estimator/ba_base.h>

#include <tbb/blocked_range.h>

namespace basalt {

template <class Scalar_>
class ScBundleAdjustmentBase : public BundleAdjustmentBase<Scalar_> {
 public:
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
  using Mat6 = Eigen::Matrix<Scalar, 6, 6>;
  using Mat63 = Eigen::Matrix<Scalar, 6, 3>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SO3 = Sophus::SO3<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;

  struct RelLinDataBase {
    std::vector<std::pair<TimeCamId, TimeCamId>> order;

    Eigen::aligned_vector<Mat6> d_rel_d_h;
    Eigen::aligned_vector<Mat6> d_rel_d_t;
  };

  struct FrameRelLinData {
    Mat6 Hpp;
    Vec6 bp;

    std::vector<int> lm_id;
    Eigen::aligned_vector<Mat63> Hpl;

    FrameRelLinData() {
      Hpp.setZero();
      bp.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct RelLinData : public RelLinDataBase {
    RelLinData(size_t num_keypoints, size_t num_rel_poses) {
      Hll.reserve(num_keypoints);
      Hllinv.reserve(num_keypoints);
      bl.reserve(num_keypoints);
      lm_to_obs.reserve(num_keypoints);

      Hpppl.reserve(num_rel_poses);
      order.reserve(num_rel_poses);

      d_rel_d_h.reserve(num_rel_poses);
      d_rel_d_t.reserve(num_rel_poses);

      error = 0;
    }

    void invert_keypoint_hessians() {
      for (const auto& [kpt_idx, hll] : Hll) {
        Mat3 Hll_inv;
        Hll_inv.setIdentity();
        // Use ldlt b/c it has good speed (no numeric indefiniteness of this 3x3
        // matrix expected), and compared ot a direct inverse (which may be even
        // faster), it can handle degenerate cases where Hll is not invertible.
        hll.ldlt().solveInPlace(Hll_inv);
        Hllinv[kpt_idx] = Hll_inv;
      }
    }

    using RelLinDataBase::d_rel_d_h;
    using RelLinDataBase::d_rel_d_t;
    using RelLinDataBase::order;

    Eigen::aligned_unordered_map<int, Mat3> Hll;
    Eigen::aligned_unordered_map<int, Mat3> Hllinv;
    Eigen::aligned_unordered_map<int, Vec3> bl;
    Eigen::aligned_unordered_map<int, std::vector<std::pair<size_t, size_t>>>
        lm_to_obs;

    Eigen::aligned_vector<FrameRelLinData> Hpppl;

    Scalar error;
  };

  struct FrameAbsLinData {
    Mat6 Hphph;
    Vec6 bph;

    Mat6 Hptpt;
    Vec6 bpt;

    Mat6 Hphpt;

    std::vector<int> lm_id;
    Eigen::aligned_vector<Mat63> Hphl;
    Eigen::aligned_vector<Mat63> Hptl;

    FrameAbsLinData() {
      Hphph.setZero();
      Hptpt.setZero();
      Hphpt.setZero();

      bph.setZero();
      bpt.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct AbsLinData {
    AbsLinData(size_t num_keypoints, size_t num_rel_poses) {
      Hll.reserve(num_keypoints);
      Hllinv.reserve(num_keypoints);
      bl.reserve(num_keypoints);
      lm_to_obs.reserve(num_keypoints);

      Hpppl.reserve(num_rel_poses);
      order.reserve(num_rel_poses);

      error = 0;
    }

    void invert_keypoint_hessians() {
      for (const auto& [kpt_idx, hll] : Hll) {
        Mat3 Hll_inv;
        Hll_inv.setIdentity();
        // Use ldlt b/c it has good speed (no numeric indefiniteness of this 3x3
        // matrix expected), and compared ot a direct inverse (which may be even
        // faster), it can handle degenerate cases where Hll is not invertible.
        hll.ldlt().solveInPlace(Hll_inv);
        Hllinv[kpt_idx] = Hll_inv;
      }
    }

    std::vector<std::pair<TimeCamId, TimeCamId>> order;

    Eigen::aligned_unordered_map<int, Mat3> Hll;
    Eigen::aligned_unordered_map<int, Mat3> Hllinv;
    Eigen::aligned_unordered_map<int, Vec3> bl;
    Eigen::aligned_unordered_map<int, std::vector<std::pair<size_t, size_t>>>
        lm_to_obs;

    Eigen::aligned_vector<FrameAbsLinData> Hpppl;

    Scalar error;
  };

  using BundleAdjustmentBase<Scalar>::computeDelta;

  void linearizeHelper(
      Eigen::aligned_vector<RelLinData>& rld_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      Scalar& error) const {
    linearizeHelperStatic(rld_vec, obs_to_lin, this, error);
  }

  void linearizeAbsHelper(
      Eigen::aligned_vector<AbsLinData>& ald_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      Scalar& error) const {
    linearizeHelperAbsStatic(ald_vec, obs_to_lin, this, error);
  }

  static void linearizeHelperStatic(
      Eigen::aligned_vector<RelLinData>& rld_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      const BundleAdjustmentBase<Scalar>* ba_base, Scalar& error);

  void linearizeHelperAbs(
      Eigen::aligned_vector<AbsLinData>& ald_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      Scalar& error) const {
    linearizeHelperAbsStatic(ald_vec, obs_to_lin, this, error);
  }

  static void linearizeHelperAbsStatic(
      Eigen::aligned_vector<AbsLinData>& ald_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      const BundleAdjustmentBase<Scalar>* ba_base, Scalar& error);

  static void linearizeRel(const RelLinData& rld, MatX& H, VecX& b);

  static void updatePoints(const AbsOrderMap& aom, const RelLinData& rld,
                           const VecX& inc, LandmarkDatabase<Scalar>& lmdb,
                           Scalar* l_diff = nullptr);

  static void updatePointsAbs(const AbsOrderMap& aom, const AbsLinData& ald,
                              const VecX& inc, LandmarkDatabase<Scalar>& lmdb,
                              Scalar* l_diff = nullptr);

  static Eigen::VectorXd checkNullspace(
      const MatX& H, const VecX& b, const AbsOrderMap& marg_order,
      const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
          frame_states,
      const Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>>& frame_poses,
      bool verbose = true);

  static Eigen::VectorXd checkEigenvalues(const MatX& H, bool verbose = true);

  static void computeImuError(
      const AbsOrderMap& aom, Scalar& imu_error, Scalar& bg_error,
      Scalar& ba_error,
      const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
          states,
      const Eigen::aligned_map<int64_t, IntegratedImuMeasurement<Scalar>>&
          imu_meas,
      const Vec3& gyro_bias_weight, const Vec3& accel_bias_weight,
      const Vec3& g);

  template <class AccumT>
  static void linearizeAbs(const MatX& rel_H, const VecX& rel_b,
                           const RelLinDataBase& rld, const AbsOrderMap& aom,
                           AccumT& accum) {
    // int asize = aom.total_size;

    //  BASALT_ASSERT(abs_H.cols() == asize);
    //  BASALT_ASSERT(abs_H.rows() == asize);
    //  BASALT_ASSERT(abs_b.rows() == asize);

    for (size_t i = 0; i < rld.order.size(); i++) {
      const TimeCamId& tcid_h = rld.order[i].first;
      const TimeCamId& tcid_ti = rld.order[i].second;

      int abs_h_idx = aom.abs_order_map.at(tcid_h.frame_id).first;
      int abs_ti_idx = aom.abs_order_map.at(tcid_ti.frame_id).first;

      accum.template addB<POSE_SIZE>(
          abs_h_idx, rld.d_rel_d_h[i].transpose() *
                         rel_b.template segment<POSE_SIZE>(i * POSE_SIZE));
      accum.template addB<POSE_SIZE>(
          abs_ti_idx, rld.d_rel_d_t[i].transpose() *
                          rel_b.template segment<POSE_SIZE>(i * POSE_SIZE));

      for (size_t j = 0; j < rld.order.size(); j++) {
        BASALT_ASSERT(rld.order[i].first == rld.order[j].first);

        const TimeCamId& tcid_tj = rld.order[j].second;

        int abs_tj_idx = aom.abs_order_map.at(tcid_tj.frame_id).first;

        if (tcid_h.frame_id == tcid_ti.frame_id ||
            tcid_h.frame_id == tcid_tj.frame_id)
          continue;

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_h_idx, abs_h_idx,
            rld.d_rel_d_h[i].transpose() *
                rel_H.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                           POSE_SIZE * j) *
                rld.d_rel_d_h[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_ti_idx, abs_h_idx,
            rld.d_rel_d_t[i].transpose() *
                rel_H.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                           POSE_SIZE * j) *
                rld.d_rel_d_h[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_h_idx, abs_tj_idx,
            rld.d_rel_d_h[i].transpose() *
                rel_H.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                           POSE_SIZE * j) *
                rld.d_rel_d_t[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_ti_idx, abs_tj_idx,
            rld.d_rel_d_t[i].transpose() *
                rel_H.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                           POSE_SIZE * j) *
                rld.d_rel_d_t[j]);
      }
    }
  }

  template <class AccumT>
  static void linearizeAbs(const AbsLinData& ald, const AbsOrderMap& aom,
                           AccumT& accum) {
    for (size_t i = 0; i < ald.order.size(); i++) {
      const TimeCamId& tcid_h = ald.order[i].first;
      const TimeCamId& tcid_ti = ald.order[i].second;

      int abs_h_idx = aom.abs_order_map.at(tcid_h.frame_id).first;
      int abs_ti_idx = aom.abs_order_map.at(tcid_ti.frame_id).first;

      const FrameAbsLinData& fald = ald.Hpppl.at(i);

      // Pose H and b part
      accum.template addH<POSE_SIZE, POSE_SIZE>(abs_h_idx, abs_h_idx,
                                                fald.Hphph);
      accum.template addH<POSE_SIZE, POSE_SIZE>(abs_ti_idx, abs_ti_idx,
                                                fald.Hptpt);
      accum.template addH<POSE_SIZE, POSE_SIZE>(abs_h_idx, abs_ti_idx,
                                                fald.Hphpt);
      accum.template addH<POSE_SIZE, POSE_SIZE>(abs_ti_idx, abs_h_idx,
                                                fald.Hphpt.transpose());

      accum.template addB<POSE_SIZE>(abs_h_idx, fald.bph);
      accum.template addB<POSE_SIZE>(abs_ti_idx, fald.bpt);

      // Schur complement for landmark part
      for (size_t j = 0; j < fald.lm_id.size(); j++) {
        Eigen::Matrix<Scalar, POSE_SIZE, 3> H_phl_H_ll_inv, H_ptl_H_ll_inv;
        int lm_id = fald.lm_id.at(j);

        H_phl_H_ll_inv = fald.Hphl[j] * ald.Hllinv.at(lm_id);
        H_ptl_H_ll_inv = fald.Hptl[j] * ald.Hllinv.at(lm_id);

        accum.template addB<POSE_SIZE>(abs_h_idx,
                                       -H_phl_H_ll_inv * ald.bl.at(lm_id));
        accum.template addB<POSE_SIZE>(abs_ti_idx,
                                       -H_ptl_H_ll_inv * ald.bl.at(lm_id));

        const auto& other_obs = ald.lm_to_obs.at(lm_id);

        for (size_t k = 0; k < other_obs.size(); k++) {
          int other_frame_idx = other_obs[k].first;
          int other_lm_idx = other_obs[k].second;
          const FrameAbsLinData& fald_other = ald.Hpppl.at(other_frame_idx);
          const TimeCamId& tcid_hk = ald.order.at(other_frame_idx).first;
          const TimeCamId& tcid_tk = ald.order.at(other_frame_idx).second;

          // Assume same host frame
          BASALT_ASSERT(tcid_hk.frame_id == tcid_h.frame_id &&
                        tcid_hk.cam_id == tcid_h.cam_id);

          int abs_tk_idx = aom.abs_order_map.at(tcid_tk.frame_id).first;

          Eigen::Matrix<Scalar, 3, POSE_SIZE> H_l_ph_other =
              fald_other.Hphl[other_lm_idx].transpose();

          Eigen::Matrix<Scalar, 3, POSE_SIZE> H_l_pt_other =
              fald_other.Hptl[other_lm_idx].transpose();

          accum.template addH<POSE_SIZE, POSE_SIZE>(
              abs_h_idx, abs_h_idx, -H_phl_H_ll_inv * H_l_ph_other);
          accum.template addH<POSE_SIZE, POSE_SIZE>(
              abs_ti_idx, abs_h_idx, -H_ptl_H_ll_inv * H_l_ph_other);
          accum.template addH<POSE_SIZE, POSE_SIZE>(
              abs_h_idx, abs_tk_idx, -H_phl_H_ll_inv * H_l_pt_other);
          accum.template addH<POSE_SIZE, POSE_SIZE>(
              abs_ti_idx, abs_tk_idx, -H_ptl_H_ll_inv * H_l_pt_other);
        }
      }
    }
  }

  template <class AccumT>
  struct LinearizeAbsReduce {
    static_assert(std::is_same_v<typename AccumT::Scalar, Scalar>);

    using RelLinConstDataIter =
        typename Eigen::aligned_vector<RelLinData>::const_iterator;

    LinearizeAbsReduce(const AbsOrderMap& aom) : aom(aom) {
      accum.reset(aom.total_size);
    }

    LinearizeAbsReduce(const LinearizeAbsReduce& other, tbb::split)
        : aom(other.aom) {
      accum.reset(aom.total_size);
    }

    void operator()(const tbb::blocked_range<RelLinConstDataIter>& range) {
      for (const RelLinData& rld : range) {
        MatX rel_H;
        VecX rel_b;
        linearizeRel(rld, rel_H, rel_b);

        linearizeAbs(rel_H, rel_b, rld, aom, accum);
      }
    }

    void join(LinearizeAbsReduce& rhs) { accum.join(rhs.accum); }

    const AbsOrderMap& aom;
    AccumT accum;
  };

  template <class AccumT>
  struct LinearizeAbsReduce2 {
    static_assert(std::is_same_v<typename AccumT::Scalar, Scalar>);

    using AbsLinDataConstIter =
        typename Eigen::aligned_vector<AbsLinData>::const_iterator;

    LinearizeAbsReduce2(const AbsOrderMap& aom) : aom(aom) {
      accum.reset(aom.total_size);
    }

    LinearizeAbsReduce2(const LinearizeAbsReduce2& other, tbb::split)
        : aom(other.aom) {
      accum.reset(aom.total_size);
    }

    void operator()(const tbb::blocked_range<AbsLinDataConstIter>& range) {
      for (const AbsLinData& ald : range) {
        linearizeAbs(ald, aom, accum);
      }
    }

    void join(LinearizeAbsReduce2& rhs) { accum.join(rhs.accum); }

    const AbsOrderMap& aom;
    AccumT accum;
  };
};
}  // namespace basalt
