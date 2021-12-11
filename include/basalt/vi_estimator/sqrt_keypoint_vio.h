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

#include <thread>

#include <basalt/imu/preintegration.h>
#include <basalt/utils/time_utils.hpp>

#include <basalt/vi_estimator/sqrt_ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <basalt/imu/preintegration.h>

namespace basalt {

template <class Scalar_>
class SqrtKeypointVioEstimator : public VioEstimatorBase,
                                 public SqrtBundleAdjustmentBase<Scalar_> {
 public:
  using Scalar = Scalar_;

  typedef std::shared_ptr<SqrtKeypointVioEstimator> Ptr;

  static const int N = 9;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using VecN = Eigen::Matrix<Scalar, N, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatN3 = Eigen::Matrix<Scalar, N, 3>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SE3 = Sophus::SE3<Scalar>;

  using typename SqrtBundleAdjustmentBase<Scalar>::RelLinData;
  using typename SqrtBundleAdjustmentBase<Scalar>::AbsLinData;

  using BundleAdjustmentBase<Scalar>::computeError;
  using BundleAdjustmentBase<Scalar>::get_current_points;
  using BundleAdjustmentBase<Scalar>::computeDelta;
  using BundleAdjustmentBase<Scalar>::computeProjections;
  using BundleAdjustmentBase<Scalar>::triangulate;
  using BundleAdjustmentBase<Scalar>::backup;
  using BundleAdjustmentBase<Scalar>::restore;
  using BundleAdjustmentBase<Scalar>::getPoseStateWithLin;
  using BundleAdjustmentBase<Scalar>::computeModelCostChange;

  using SqrtBundleAdjustmentBase<Scalar>::linearizeHelper;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeAbsHelper;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeRel;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeAbs;
  using SqrtBundleAdjustmentBase<Scalar>::updatePoints;
  using SqrtBundleAdjustmentBase<Scalar>::updatePointsAbs;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeMargPrior;
  using SqrtBundleAdjustmentBase<Scalar>::computeMargPriorError;
  using SqrtBundleAdjustmentBase<Scalar>::computeMargPriorModelCostChange;
  using SqrtBundleAdjustmentBase<Scalar>::checkNullspace;
  using SqrtBundleAdjustmentBase<Scalar>::checkEigenvalues;

  SqrtKeypointVioEstimator(const Eigen::Vector3d& g,
                           const basalt::Calibration<double>& calib,
                           const VioConfig& config);

  void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                  const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  void initialize(const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  virtual ~SqrtKeypointVioEstimator() { maybe_join(); }

  inline void maybe_join() override {
    if (processing_thread) {
      processing_thread->join();
      processing_thread.reset();
    }
  }

  void addIMUToQueue(const ImuData<double>::Ptr& data) override;
  void addVisionToQueue(const OpticalFlowResult::Ptr& data) override;

  typename ImuData<Scalar>::Ptr popFromImuDataQueue();

  bool measure(const OpticalFlowResult::Ptr& opt_flow_meas,
               const typename IntegratedImuMeasurement<Scalar>::Ptr& meas);

  // int64_t propagate();
  // void addNewState(int64_t data_t_ns);

  void optimize_and_marg(const std::map<int64_t, int>& num_points_connected,
                         const std::unordered_set<KeypointId>& lost_landmaks);

  void marginalize(const std::map<int64_t, int>& num_points_connected,
                   const std::unordered_set<KeypointId>& lost_landmaks);
  void optimize();

  void debug_finalize() override;

  void logMargNullspace();
  Eigen::VectorXd checkMargNullspace() const;
  Eigen::VectorXd checkMargEigenvalues() const;

  int64_t get_t_ns() const {
    return frame_states.at(last_state_t_ns).getState().t_ns;
  }
  const SE3& get_T_w_i() const {
    return frame_states.at(last_state_t_ns).getState().T_w_i;
  }
  const Vec3& get_vel_w_i() const {
    return frame_states.at(last_state_t_ns).getState().vel_w_i;
  }

  const PoseVelBiasState<Scalar>& get_state() const {
    return frame_states.at(last_state_t_ns).getState();
  }
  PoseVelBiasState<Scalar> get_state(int64_t t_ns) const {
    PoseVelBiasState<Scalar> state;

    auto it = frame_states.find(t_ns);

    if (it != frame_states.end()) {
      return it->second.getState();
    }

    auto it2 = frame_poses.find(t_ns);
    if (it2 != frame_poses.end()) {
      state.T_w_i = it2->second.getPose();
    }

    return state;
  }

  void setMaxStates(size_t val) override { max_states = val; }
  void setMaxKfs(size_t val) override { max_kfs = val; }

  Eigen::aligned_vector<SE3> getFrameStates() const {
    Eigen::aligned_vector<SE3> res;

    for (const auto& kv : frame_states) {
      res.push_back(kv.second.getState().T_w_i);
    }

    return res;
  }

  Eigen::aligned_vector<SE3> getFramePoses() const {
    Eigen::aligned_vector<SE3> res;

    for (const auto& kv : frame_poses) {
      res.push_back(kv.second.getPose());
    }

    return res;
  }

  Eigen::aligned_map<int64_t, SE3> getAllPosesMap() const {
    Eigen::aligned_map<int64_t, SE3> res;

    for (const auto& kv : frame_poses) {
      res[kv.first] = kv.second.getPose();
    }

    for (const auto& kv : frame_states) {
      res[kv.first] = kv.second.getState().T_w_i;
    }

    return res;
  }

  Sophus::SE3d getT_w_i_init() override {
    return T_w_i_init.template cast<double>();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  using BundleAdjustmentBase<Scalar>::frame_poses;
  using BundleAdjustmentBase<Scalar>::frame_states;
  using BundleAdjustmentBase<Scalar>::lmdb;
  using BundleAdjustmentBase<Scalar>::obs_std_dev;
  using BundleAdjustmentBase<Scalar>::huber_thresh;
  using BundleAdjustmentBase<Scalar>::calib;

 private:
  bool take_kf;
  int frames_after_kf;
  std::set<int64_t> kf_ids;

  int64_t last_state_t_ns;
  Eigen::aligned_map<int64_t, IntegratedImuMeasurement<Scalar>> imu_meas;

  const Vec3 g;

  // Input

  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

  std::map<int64_t, int> num_points_kf;

  // Marginalization
  MargLinData<Scalar> marg_data;

  // Used only for debug and log purporses.
  MargLinData<Scalar> nullspace_marg_data;

  Vec3 gyro_bias_sqrt_weight, accel_bias_sqrt_weight;

  size_t max_states;
  size_t max_kfs;

  SE3 T_w_i_init;

  bool initialized;
  bool opt_started;

  VioConfig config;

  constexpr static Scalar vee_factor = Scalar(2.0);
  constexpr static Scalar initial_vee = Scalar(2.0);
  Scalar lambda, min_lambda, max_lambda, lambda_vee;

  std::shared_ptr<std::thread> processing_thread;

  // timing and stats
  ExecutionStats stats_all_;
  ExecutionStats stats_sums_;
};
}  // namespace basalt
