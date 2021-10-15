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
#include <basalt/vi_estimator/sc_ba_base.h>

namespace basalt {

template <class Scalar_>
class SqrtBundleAdjustmentBase : public BundleAdjustmentBase<Scalar_> {
 public:
  using Scalar = Scalar_;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SO3 = Sophus::SO3<Scalar>;

  using RelLinDataBase =
      typename ScBundleAdjustmentBase<Scalar>::RelLinDataBase;
  using FrameRelLinData =
      typename ScBundleAdjustmentBase<Scalar>::FrameRelLinData;
  using RelLinData = typename ScBundleAdjustmentBase<Scalar>::RelLinData;

  using FrameAbsLinData =
      typename ScBundleAdjustmentBase<Scalar>::FrameAbsLinData;
  using AbsLinData = typename ScBundleAdjustmentBase<Scalar>::AbsLinData;

  using BundleAdjustmentBase<Scalar>::computeDelta;

  void linearizeHelper(
      Eigen::aligned_vector<RelLinData>& rld_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      Scalar& error) const {
    ScBundleAdjustmentBase<Scalar>::linearizeHelperStatic(rld_vec, obs_to_lin,
                                                          this, error);
  }

  void linearizeAbsHelper(
      Eigen::aligned_vector<AbsLinData>& ald_vec,
      const std::unordered_map<
          TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obs_to_lin,
      Scalar& error) const {
    ScBundleAdjustmentBase<Scalar>::linearizeHelperAbsStatic(
        ald_vec, obs_to_lin, this, error);
  }

  static void linearizeRel(const RelLinData& rld, MatX& H, VecX& b) {
    ScBundleAdjustmentBase<Scalar>::linearizeRel(rld, H, b);
  }

  static void updatePoints(const AbsOrderMap& aom, const RelLinData& rld,
                           const VecX& inc, LandmarkDatabase<Scalar>& lmdb,
                           Scalar* l_diff = nullptr) {
    ScBundleAdjustmentBase<Scalar>::updatePoints(aom, rld, inc, lmdb, l_diff);
  }

  static void updatePointsAbs(const AbsOrderMap& aom, const AbsLinData& ald,
                              const VecX& inc, LandmarkDatabase<Scalar>& lmdb,
                              Scalar* l_diff = nullptr) {
    ScBundleAdjustmentBase<Scalar>::updatePointsAbs(aom, ald, inc, lmdb,
                                                    l_diff);
  }

  static Eigen::VectorXd checkNullspace(
      const MargLinData<Scalar_>& mld,
      const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
          frame_states,
      const Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>>& frame_poses,
      bool verbose = true);

  static Eigen::VectorXd checkEigenvalues(const MargLinData<Scalar_>& mld,
                                          bool verbose = true);

  template <class AccumT>
  static void linearizeAbs(const MatX& rel_H, const VecX& rel_b,
                           const RelLinDataBase& rld, const AbsOrderMap& aom,
                           AccumT& accum) {
    return ScBundleAdjustmentBase<Scalar>::template linearizeAbs<AccumT>(
        rel_H, rel_b, rld, aom, accum);
  }

  template <class AccumT>
  static void linearizeAbs(const AbsLinData& ald, const AbsOrderMap& aom,
                           AccumT& accum) {
    return ScBundleAdjustmentBase<Scalar>::template linearizeAbs<AccumT>(
        ald, aom, accum);
  }
};
}  // namespace basalt
