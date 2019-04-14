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

#include <basalt/imu/preintegration.h>
#include <basalt/io/dataset_io.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/test_utils.h>
#include <basalt/camera/generic_camera.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>

#include <basalt/vi_estimator/ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

namespace basalt {

class KeypointVioEstimator : public VioEstimatorBase,
                             public BundleAdjustmentBase {
 public:
  typedef std::shared_ptr<KeypointVioEstimator> Ptr;

  static const int N = 9;
  typedef Eigen::Matrix<double, N, 1> VecN;
  typedef Eigen::Matrix<double, N, N> MatNN;
  typedef Eigen::Matrix<double, N, 3> MatN3;

  static constexpr double prior_weight = 1e8;

  KeypointVioEstimator(int64_t t_ns, const Sophus::SE3d& T_w_i,
                       const Eigen::Vector3d& vel_w_i,
                       const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                       double int_std_dev, const Eigen::Vector3d& g,
                       const basalt::Calibration<double>& calib,
                       const VioConfig& config);

  ~KeypointVioEstimator() { processing_thread->join(); }

  void addIMUToQueue(const ImuData::Ptr& data);
  void addVisionToQueue(const OpticalFlowResult::Ptr& data);

  bool measure(const OpticalFlowResult::Ptr& data,
               const IntegratedImuMeasurement::Ptr& meas);

  static void linearizeAbsIMU(
      const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
      double& imu_error, double& bg_error, double& ba_error,
      const Eigen::map<int64_t, PoseVelBiasStateWithLin>& states,
      const Eigen::map<int64_t, IntegratedImuMeasurement>& imu_meas,
      const Eigen::Vector3d& gyro_bias_weight,
      const Eigen::Vector3d& accel_bias_weight, const Eigen::Vector3d& g);

  static void computeImuError(
      const AbsOrderMap& aom, double& imu_error, double& bg_error,
      double& ba_error,
      const Eigen::map<int64_t, PoseVelBiasStateWithLin>& states,
      const Eigen::map<int64_t, IntegratedImuMeasurement>& imu_meas,
      const Eigen::Vector3d& gyro_bias_weight,
      const Eigen::Vector3d& accel_bias_weight, const Eigen::Vector3d& g);

  void linearizeMargPrior(const AbsOrderMap& aom, Eigen::MatrixXd& abs_H,
                          Eigen::VectorXd& abs_b,
                          double& marg_prior_error) const;

  void computeMargPriorError(double& marg_prior_error) const;

  void computeDelta(const AbsOrderMap& marg_order,
                    Eigen::VectorXd& delta) const;

  // int64_t propagate();
  // void addNewState(int64_t data_t_ns);

  void marginalize(const std::map<int64_t, int>& num_points_connected);

  void optimize();

  void checkMargNullspace() const;
  static Eigen::VectorXd checkNullspace(
      const Eigen::MatrixXd& marg_H, const Eigen::VectorXd& marg_b,
      const AbsOrderMap& marg_order,
      const Eigen::map<int64_t, PoseVelBiasStateWithLin>& frame_states,
      const Eigen::map<int64_t, PoseStateWithLin>& frame_poses);

  int64_t get_t_ns() const {
    return frame_states.at(last_state_t_ns).getState().t_ns;
  }
  const Sophus::SE3d& get_T_w_i() const {
    return frame_states.at(last_state_t_ns).getState().T_w_i;
  }
  const Eigen::Vector3d& get_vel_w_i() const {
    return frame_states.at(last_state_t_ns).getState().vel_w_i;
  }

  const PoseVelBiasState& get_state() const {
    return frame_states.at(last_state_t_ns).getState();
  }
  PoseVelBiasState get_state(int64_t t_ns) const {
    PoseVelBiasState state;

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
  // const MatNN get_cov() const { return cov.bottomRightCorner<N, N>(); }

  void computeProjections(
      std::vector<Eigen::vector<Eigen::Vector4d>>& res) const;

  inline void setMaxStates(size_t val) { max_states = val; }
  inline void setMaxKfs(size_t val) { max_kfs = val; }

  Eigen::vector<Sophus::SE3d> getFrameStates() const {
    Eigen::vector<Sophus::SE3d> res;

    for (const auto& kv : frame_states) {
      res.push_back(kv.second.getState().T_w_i);
    }

    return res;
  }

  Eigen::vector<Sophus::SE3d> getFramePoses() const {
    Eigen::vector<Sophus::SE3d> res;

    for (const auto& kv : frame_poses) {
      res.push_back(kv.second.getPose());
    }

    return res;
  }

  Eigen::map<int64_t, Sophus::SE3d> getAllPosesMap() const {
    Eigen::map<int64_t, Sophus::SE3d> res;

    for (const auto& kv : frame_poses) {
      res[kv.first] = kv.second.getPose();
    }

    for (const auto& kv : frame_states) {
      res[kv.first] = kv.second.getState().T_w_i;
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  bool take_kf;
  int frames_after_kf;
  std::set<int64_t> kf_ids;

  int64_t last_state_t_ns;
  Eigen::map<int64_t, IntegratedImuMeasurement> imu_meas;

  const Eigen::Vector3d g;

  // Input

  Eigen::map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

  std::map<int64_t, int> num_points_kf;

  // Marginalization
  AbsOrderMap marg_order;
  Eigen::MatrixXd marg_H;
  Eigen::VectorXd marg_b;

  Eigen::Vector3d gyro_bias_weight, accel_bias_weight;

  size_t max_states;
  size_t max_kfs;

  bool opt_started;

  VioConfig config;

  int64_t msckf_kf_id;

  std::shared_ptr<std::thread> processing_thread;
};
}  // namespace basalt
