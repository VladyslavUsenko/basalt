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

#include <basalt/vi_estimator/landmark_database.h>

namespace basalt {

template <class Scalar_>
class BundleAdjustmentBase {
 public:
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
  using Mat6 = Eigen::Matrix<Scalar, 6, 6>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  using SE3 = Sophus::SE3<Scalar>;

  void computeError(Scalar& error,
                    std::map<int, std::vector<std::pair<TimeCamId, Scalar>>>*
                        outliers = nullptr,
                    Scalar outlier_threshold = 0) const;

  void filterOutliers(Scalar outlier_threshold, int min_num_obs);

  void optimize_single_frame_pose(
      PoseStateWithLin<Scalar>& state_t,
      const std::vector<std::vector<int>>& connected_obs) const;

  template <class Scalar2>
  void get_current_points(
      Eigen::aligned_vector<Eigen::Matrix<Scalar2, 3, 1>>& points,
      std::vector<int>& ids) const;

  void computeDelta(const AbsOrderMap& marg_order, VecX& delta) const;

  void linearizeMargPrior(const MargLinData<Scalar>& mld,
                          const AbsOrderMap& aom, MatX& abs_H, VecX& abs_b,
                          Scalar& marg_prior_error) const;

  void computeMargPriorError(const MargLinData<Scalar>& mld,
                             Scalar& marg_prior_error) const;

  Scalar computeMargPriorModelCostChange(const MargLinData<Scalar>& mld,
                                         const VecX& marg_scaling,
                                         const VecX& marg_pose_inc) const;

  // TODO: Old version for squared H and b. Remove when no longer needed.
  Scalar computeModelCostChange(const MatX& H, const VecX& b,
                                const VecX& inc) const;

  template <class Scalar2>
  void computeProjections(
      std::vector<Eigen::aligned_vector<Eigen::Matrix<Scalar2, 4, 1>>>& data,
      FrameId last_state_t_ns) const;

  /// Triangulates the point and returns homogenous representation. First 3
  /// components - unit-length direction vector. Last component inverse
  /// distance.
  template <class Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 1> triangulate(
      const Eigen::MatrixBase<Derived>& f0,
      const Eigen::MatrixBase<Derived>& f1,
      const Sophus::SE3<typename Derived::Scalar>& T_0_1) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

    // suffix "2" to avoid name clash with class-wide typedefs
    using Scalar_2 = typename Derived::Scalar;
    using Vec4_2 = Eigen::Matrix<Scalar_2, 4, 1>;

    Eigen::Matrix<Scalar_2, 3, 4> P1, P2;
    P1.setIdentity();
    P2 = T_0_1.inverse().matrix3x4();

    Eigen::Matrix<Scalar_2, 4, 4> A(4, 4);
    A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
    A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
    A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
    A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix<Scalar_2, 4, 4>> mySVD(A,
                                                          Eigen::ComputeFullV);
    Vec4_2 worldPoint = mySVD.matrixV().col(3);
    worldPoint /= worldPoint.template head<3>().norm();

    // Enforce same direction of bearing vector and initial point
    if (f0.dot(worldPoint.template head<3>()) < 0) worldPoint *= -1;

    return worldPoint;
  }

  inline void backup() {
    for (auto& kv : frame_states) kv.second.backup();
    for (auto& kv : frame_poses) kv.second.backup();
    lmdb.backup();
  }

  inline void restore() {
    for (auto& kv : frame_states) kv.second.restore();
    for (auto& kv : frame_poses) kv.second.restore();
    lmdb.restore();
  }

  // protected:
  PoseStateWithLin<Scalar> getPoseStateWithLin(int64_t t_ns) const {
    auto it = frame_poses.find(t_ns);
    if (it != frame_poses.end()) return it->second;

    auto it2 = frame_states.find(t_ns);
    if (it2 == frame_states.end()) {
      std::cerr << "Could not find pose " << t_ns << std::endl;
      std::abort();
    }

    return PoseStateWithLin<Scalar>(it2->second);
  }

  Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>> frame_states;
  Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>> frame_poses;

  // Point management
  LandmarkDatabase<Scalar> lmdb;

  Scalar obs_std_dev;
  Scalar huber_thresh;

  basalt::Calibration<Scalar> calib;
};
}  // namespace basalt
