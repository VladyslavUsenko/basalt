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

#include <basalt/utils/imu_types.h>

#include <tbb/blocked_range.h>

namespace basalt {

class BundleAdjustmentBase {
 public:
  // keypoint position defined relative to some frame
  struct KeypointPosition {
    TimeCamId kf_id;
    Eigen::Vector2d dir;
    double id;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct KeypointObservation {
    int kpt_id;
    Eigen::Vector2d pos;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct RelLinDataBase {
    std::vector<std::pair<TimeCamId, TimeCamId>> order;

    Eigen::vector<Sophus::Matrix6d> d_rel_d_h;
    Eigen::vector<Sophus::Matrix6d> d_rel_d_t;
  };

  struct FrameRelLinData {
    Sophus::Matrix6d Hpp;
    Sophus::Vector6d bp;

    std::vector<int> lm_id;
    Eigen::vector<Eigen::Matrix<double, 6, 3>> Hpl;

    FrameRelLinData() {
      Hpp.setZero();
      bp.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct RelLinData : public RelLinDataBase {
    RelLinData(size_t num_keypoints, size_t num_rel_poses) {
      Hll.reserve(num_keypoints);
      bl.reserve(num_keypoints);
      lm_to_obs.reserve(num_keypoints);

      Hpppl.reserve(num_rel_poses);
      order.reserve(num_rel_poses);

      d_rel_d_h.reserve(num_rel_poses);
      d_rel_d_t.reserve(num_rel_poses);

      error = 0;
    }

    void invert_keypoint_hessians() {
      for (auto& kv : Hll) {
        Eigen::Matrix3d Hll_inv;
        Hll_inv.setIdentity();
        kv.second.ldlt().solveInPlace(Hll_inv);
        kv.second = Hll_inv;
      }
    }

    Eigen::unordered_map<int, Eigen::Matrix3d> Hll;
    Eigen::unordered_map<int, Eigen::Vector3d> bl;
    Eigen::unordered_map<int, std::vector<std::pair<size_t, size_t>>> lm_to_obs;

    Eigen::vector<FrameRelLinData> Hpppl;

    double error;
  };

  void computeError(double& error) const;

  void linearizeHelper(
      Eigen::vector<RelLinData>& rld_vec,
      const Eigen::map<
          TimeCamId, Eigen::map<TimeCamId, Eigen::vector<KeypointObservation>>>&
          obs_to_lin,
      double& error) const;

  static void linearizeRel(const RelLinData& rld, Eigen::MatrixXd& H,
                           Eigen::VectorXd& b);

  template <class CamT>
  static bool linearizePoint(
      const KeypointObservation& kpt_obs, const KeypointPosition& kpt_pos,
      const Eigen::Matrix4d& T_t_h, const CamT& cam, Eigen::Vector2d& res,
      Eigen::Matrix<double, 2, POSE_SIZE>* d_res_d_xi = nullptr,
      Eigen::Matrix<double, 2, 3>* d_res_d_p = nullptr,
      Eigen::Vector4d* proj = nullptr) {
    // Todo implement without jacobians
    Eigen::Matrix<double, 4, 2> Jup;
    Eigen::Vector4d p_h_3d;
    p_h_3d = StereographicParam<double>::unproject(kpt_pos.dir, &Jup);
    p_h_3d[3] = kpt_pos.id;

    Eigen::Vector4d p_t_3d = T_t_h * p_h_3d;

    Eigen::Matrix<double, 4, POSE_SIZE> d_point_d_xi;
    d_point_d_xi.topLeftCorner<3, 3>() =
        Eigen::Matrix3d::Identity() * kpt_pos.id;
    d_point_d_xi.topRightCorner<3, 3>() = -Sophus::SO3d::hat(p_t_3d.head<3>());
    d_point_d_xi.row(3).setZero();

    Eigen::Matrix<double, 2, 4> Jp;
    bool valid = cam.project(p_t_3d, res, &Jp);

    if (!valid) {
      //      std::cerr << " Invalid projection! kpt_pos.dir "
      //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
      //                kpt_pos.id
      //                << " idx " << kpt_obs.kpt_id << std::endl;

      //      std::cerr << "T_t_h\n" << T_t_h << std::endl;
      //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;
      //      std::cerr << "p_t_3d\n" << p_t_3d.transpose() << std::endl;

      return false;
    }

    if (proj) {
      proj->head<2>() = res;
      (*proj)[2] = p_t_3d[3] / p_t_3d.head<3>().norm();
    }
    res -= kpt_obs.pos;

    if (d_res_d_xi) {
      *d_res_d_xi = Jp * d_point_d_xi;
    }

    if (d_res_d_p) {
      Eigen::Matrix<double, 4, 3> Jpp;
      Jpp.block<3, 2>(0, 0) = T_t_h.topLeftCorner<3, 4>() * Jup;
      Jpp.col(2) = T_t_h.col(3);

      *d_res_d_p = Jp * Jpp;
    }

    return true;
  }

  template <class CamT>
  inline static bool linearizePoint(
      const KeypointObservation& kpt_obs, const KeypointPosition& kpt_pos,
      const CamT& cam, Eigen::Vector2d& res,
      Eigen::Matrix<double, 2, 3>* d_res_d_p = nullptr,
      Eigen::Vector4d* proj = nullptr) {
    // Todo implement without jacobians
    Eigen::Matrix<double, 4, 2> Jup;
    Eigen::Vector4d p_h_3d;
    p_h_3d = StereographicParam<double>::unproject(kpt_pos.dir, &Jup);

    Eigen::Matrix<double, 2, 4> Jp;
    bool valid = cam.project(p_h_3d, res, &Jp);

    if (!valid) {
      //      std::cerr << " Invalid projection! kpt_pos.dir "
      //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
      //                kpt_pos.id
      //                << " idx " << kpt_obs.kpt_id << std::endl;
      //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;

      return false;
    }

    if (proj) {
      proj->head<2>() = res;
      (*proj)[2] = kpt_pos.id;
    }
    res -= kpt_obs.pos;

    if (d_res_d_p) {
      Eigen::Matrix<double, 4, 3> Jpp;
      Jpp.block<4, 2>(0, 0) = Jup;
      Jpp.col(2).setZero();

      *d_res_d_p = Jp * Jpp;
    }

    return true;
  }

  void updatePoints(const AbsOrderMap& aom, const RelLinData& rld,
                    const Eigen::VectorXd& inc);

  static Sophus::SE3d computeRelPose(const Sophus::SE3d& T_w_i_h,
                                     const Sophus::SE3d& T_w_i_t,
                                     const Sophus::SE3d& T_i_c_h,
                                     const Sophus::SE3d& T_i_c_t,
                                     Sophus::Matrix6d* d_rel_d_h = nullptr,
                                     Sophus::Matrix6d* d_rel_d_t = nullptr);

  void get_current_points(Eigen::vector<Eigen::Vector3d>& points,
                          std::vector<int>& ids) const;

  // Modifies abs_H and abs_b as a side effect.
  static void marginalizeHelper(Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
                                const std::set<int>& idx_to_keep,
                                const std::set<int>& idx_to_marg,
                                Eigen::MatrixXd& marg_H,
                                Eigen::VectorXd& marg_b);

  static Eigen::Vector4d triangulate(const Eigen::Vector3d& p0_3d,
                                     const Eigen::Vector3d& p1_3d,
                                     const Sophus::SE3d& T_0_1);

  template <class AccumT>
  static void linearizeAbs(const Eigen::MatrixXd& rel_H,
                           const Eigen::VectorXd& rel_b,
                           const RelLinDataBase& rld, const AbsOrderMap& aom,
                           AccumT& accum) {
    // int asize = aom.total_size;

    //  BASALT_ASSERT(abs_H.cols() == asize);
    //  BASALT_ASSERT(abs_H.rows() == asize);
    //  BASALT_ASSERT(abs_b.rows() == asize);

    for (size_t i = 0; i < rld.order.size(); i++) {
      const TimeCamId& tcid_h = rld.order[i].first;
      const TimeCamId& tcid_ti = rld.order[i].second;

      int abs_h_idx = aom.abs_order_map.at(tcid_h.first).first;
      int abs_ti_idx = aom.abs_order_map.at(tcid_ti.first).first;

      accum.template addB<POSE_SIZE>(
          abs_h_idx, rld.d_rel_d_h[i].transpose() *
                         rel_b.segment<POSE_SIZE>(i * POSE_SIZE));
      accum.template addB<POSE_SIZE>(
          abs_ti_idx, rld.d_rel_d_t[i].transpose() *
                          rel_b.segment<POSE_SIZE>(i * POSE_SIZE));

      for (size_t j = 0; j < rld.order.size(); j++) {
        BASALT_ASSERT(rld.order[i].first == rld.order[j].first);

        const TimeCamId& tcid_tj = rld.order[j].second;

        int abs_tj_idx = aom.abs_order_map.at(tcid_tj.first).first;

        if (tcid_h.first == tcid_ti.first || tcid_h.first == tcid_tj.first)
          continue;

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_h_idx, abs_h_idx, rld.d_rel_d_h[i].transpose() *
                                      rel_H.block<POSE_SIZE, POSE_SIZE>(
                                          POSE_SIZE * i, POSE_SIZE * j) *
                                      rld.d_rel_d_h[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_ti_idx, abs_h_idx, rld.d_rel_d_t[i].transpose() *
                                       rel_H.block<POSE_SIZE, POSE_SIZE>(
                                           POSE_SIZE * i, POSE_SIZE * j) *
                                       rld.d_rel_d_h[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_h_idx, abs_tj_idx, rld.d_rel_d_h[i].transpose() *
                                       rel_H.block<POSE_SIZE, POSE_SIZE>(
                                           POSE_SIZE * i, POSE_SIZE * j) *
                                       rld.d_rel_d_t[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_ti_idx, abs_tj_idx, rld.d_rel_d_t[i].transpose() *
                                        rel_H.block<POSE_SIZE, POSE_SIZE>(
                                            POSE_SIZE * i, POSE_SIZE * j) *
                                        rld.d_rel_d_t[j]);
      }
    }
  }

  template <class AccumT>
  struct LinearizeAbsReduce {
    using RelLinDataIter = Eigen::vector<RelLinData>::iterator;

    LinearizeAbsReduce(AbsOrderMap& aom) : aom(aom) {
      accum.reset(aom.total_size);
    }

    LinearizeAbsReduce(const LinearizeAbsReduce& other, tbb::split)
        : aom(other.aom) {
      accum.reset(aom.total_size);
    }

    void operator()(const tbb::blocked_range<RelLinDataIter>& range) {
      for (RelLinData& rld : range) {
        rld.invert_keypoint_hessians();

        Eigen::MatrixXd rel_H;
        Eigen::VectorXd rel_b;
        linearizeRel(rld, rel_H, rel_b);

        linearizeAbs(rel_H, rel_b, rld, aom, accum);
      }
    }

    void join(LinearizeAbsReduce& rhs) { accum.join(rhs.accum); }

    AbsOrderMap& aom;
    AccumT accum;
  };

  // protected:
  PoseStateWithLin getPoseStateWithLin(int64_t t_ns) const {
    auto it = frame_poses.find(t_ns);
    if (it != frame_poses.end()) return it->second;

    auto it2 = frame_states.find(t_ns);
    if (it2 == frame_states.end()) {
      std::cerr << "Could not find pose " << t_ns << std::endl;
      std::abort();
    }

    return PoseStateWithLin(it2->second);
  }

  Eigen::map<int64_t, PoseVelBiasStateWithLin> frame_states;
  Eigen::map<int64_t, PoseStateWithLin> frame_poses;

  // Point management
  Eigen::unordered_map<int, KeypointPosition> kpts;
  Eigen::map<TimeCamId,
             Eigen::map<TimeCamId, Eigen::vector<KeypointObservation>>>
      obs;

  double obs_std_dev;
  double huber_thresh;

  basalt::Calibration<double> calib;
};
}
