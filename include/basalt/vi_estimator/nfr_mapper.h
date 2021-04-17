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

#include <memory>
#include <thread>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <basalt/utils/common_types.h>
#include <basalt/vi_estimator/ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace basalt {

template <size_t N>
class HashBow;

class NfrMapper : public BundleAdjustmentBase {
 public:
  using Ptr = std::shared_ptr<NfrMapper>;

  template <class AccumT>
  struct MapperLinearizeAbsReduce
      : public BundleAdjustmentBase::LinearizeAbsReduce<AccumT> {
    using RollPitchFactorConstIter =
        Eigen::aligned_vector<RollPitchFactor>::const_iterator;
    using RelPoseFactorConstIter =
        Eigen::aligned_vector<RelPoseFactor>::const_iterator;
    using RelLinDataIter = Eigen::aligned_vector<RelLinData>::iterator;

    MapperLinearizeAbsReduce(
        AbsOrderMap& aom,
        const Eigen::aligned_map<int64_t, PoseStateWithLin<double>>*
            frame_poses)
        : BundleAdjustmentBase::LinearizeAbsReduce<AccumT>(aom),
          frame_poses(frame_poses) {
      this->accum.reset(aom.total_size);
      roll_pitch_error = 0;
      rel_error = 0;
    }

    MapperLinearizeAbsReduce(const MapperLinearizeAbsReduce& other, tbb::split)
        : BundleAdjustmentBase::LinearizeAbsReduce<AccumT>(other.aom),
          frame_poses(other.frame_poses) {
      this->accum.reset(this->aom.total_size);
      roll_pitch_error = 0;
      rel_error = 0;
    }

    void join(MapperLinearizeAbsReduce& rhs) {
      this->accum.join(rhs.accum);
      roll_pitch_error += rhs.roll_pitch_error;
      rel_error += rhs.rel_error;
    }

    void operator()(const tbb::blocked_range<RelLinDataIter>& range) {
      for (RelLinData& rld : range) {
        rld.invert_keypoint_hessians();

        Eigen::MatrixXd rel_H;
        Eigen::VectorXd rel_b;
        linearizeRel(rld, rel_H, rel_b);

        linearizeAbs(rel_H, rel_b, rld, this->aom, this->accum);
      }
    }

    void operator()(const tbb::blocked_range<RollPitchFactorConstIter>& range) {
      for (const RollPitchFactor& rpf : range) {
        const Sophus::SE3d& pose = frame_poses->at(rpf.t_ns).getPose();

        int idx = this->aom.abs_order_map.at(rpf.t_ns).first;

        Eigen::Matrix<double, 2, POSE_SIZE> J;
        Sophus::Vector2d res = basalt::rollPitchError(pose, rpf.R_w_i_meas, &J);

        this->accum.template addH<POSE_SIZE, POSE_SIZE>(
            idx, idx, J.transpose() * rpf.cov_inv * J);
        this->accum.template addB<POSE_SIZE>(idx,
                                             J.transpose() * rpf.cov_inv * res);

        roll_pitch_error += res.transpose() * rpf.cov_inv * res;
      }
    }

    void operator()(const tbb::blocked_range<RelPoseFactorConstIter>& range) {
      for (const RelPoseFactor& rpf : range) {
        const Sophus::SE3d& pose_i = frame_poses->at(rpf.t_i_ns).getPose();
        const Sophus::SE3d& pose_j = frame_poses->at(rpf.t_j_ns).getPose();

        int idx_i = this->aom.abs_order_map.at(rpf.t_i_ns).first;
        int idx_j = this->aom.abs_order_map.at(rpf.t_j_ns).first;

        Sophus::Matrix6d Ji, Jj;
        Sophus::Vector6d res =
            basalt::relPoseError(rpf.T_i_j, pose_i, pose_j, &Ji, &Jj);

        this->accum.template addH<POSE_SIZE, POSE_SIZE>(
            idx_i, idx_i, Ji.transpose() * rpf.cov_inv * Ji);
        this->accum.template addH<POSE_SIZE, POSE_SIZE>(
            idx_i, idx_j, Ji.transpose() * rpf.cov_inv * Jj);
        this->accum.template addH<POSE_SIZE, POSE_SIZE>(
            idx_j, idx_i, Jj.transpose() * rpf.cov_inv * Ji);
        this->accum.template addH<POSE_SIZE, POSE_SIZE>(
            idx_j, idx_j, Jj.transpose() * rpf.cov_inv * Jj);

        this->accum.template addB<POSE_SIZE>(
            idx_i, Ji.transpose() * rpf.cov_inv * res);
        this->accum.template addB<POSE_SIZE>(
            idx_j, Jj.transpose() * rpf.cov_inv * res);

        rel_error += res.transpose() * rpf.cov_inv * res;
      }
    }

    double roll_pitch_error;
    double rel_error;

    const Eigen::aligned_map<int64_t, PoseStateWithLin<double>>* frame_poses;
  };

  NfrMapper(const basalt::Calibration<double>& calib, const VioConfig& config);

  void addMargData(basalt::MargData::Ptr& data);

  void processMargData(basalt::MargData& m);

  bool extractNonlinearFactors(basalt::MargData& m);

  void optimize(int num_iterations = 10);

  Eigen::aligned_map<int64_t, PoseStateWithLin<double>>& getFramePoses();

  void computeRelPose(double& rel_error);

  void computeRollPitch(double& roll_pitch_error);

  void detect_keypoints();

  // Feature matching and inlier filtering for stereo pairs with known pose
  void match_stereo();

  void match_all();

  void build_tracks();

  void setup_opt();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::aligned_vector<RollPitchFactor> roll_pitch_factors;
  Eigen::aligned_vector<RelPoseFactor> rel_pose_factors;

  std::unordered_map<int64_t, OpticalFlowInput::Ptr> img_data;

  Corners feature_corners;

  Matches feature_matches;

  FeatureTracks feature_tracks;

  std::shared_ptr<HashBow<256>> hash_bow_database;

  VioConfig config;

  double lambda, min_lambda, max_lambda, lambda_vee;
};
}  // namespace basalt
