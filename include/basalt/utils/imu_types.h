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

#include <atomic>
#include <map>
#include <memory>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <basalt/imu/imu_types.h>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/utils/common_types.h>
#include <basalt/utils/sophus_utils.hpp>

#include <cereal/cereal.hpp>

namespace basalt {

namespace constants {
static const Eigen::Vector3d g(0, 0, -9.81);
static const Eigen::Vector3d g_dir(0, 0, -1);
}  // namespace constants

struct PoseVelBiasStateWithLin {
  using VecN = PoseVelBiasState::VecN;

  PoseVelBiasStateWithLin() {
    linearized = false;
    delta.setZero();
  };

  PoseVelBiasStateWithLin(int64_t t_ns, const Sophus::SE3d& T_w_i,
                          const Eigen::Vector3d& vel_w_i,
                          const Eigen::Vector3d& bias_gyro,
                          const Eigen::Vector3d& bias_accel, bool linearized)
      : linearized(linearized),
        state_linearized(t_ns, T_w_i, vel_w_i, bias_gyro, bias_accel) {
    delta.setZero();
    state_current = state_linearized;
  }

  PoseVelBiasStateWithLin(const PoseVelBiasState& other)
      : linearized(false), state_linearized(other) {
    delta.setZero();
    state_current = other;
  }

  void setLinFalse() {
    linearized = false;
    delta.setZero();
  }

  void setLinTrue() {
    linearized = true;
    BASALT_ASSERT(delta.isApproxToConstant(0));
    state_current = state_linearized;
  }

  void applyInc(const VecN& inc) {
    if (!linearized) {
      state_linearized.applyInc(inc);
    } else {
      delta += inc;
      state_current = state_linearized;
      state_current.applyInc(delta);
    }
  }

  inline const PoseVelBiasState& getState() const {
    if (!linearized) {
      return state_linearized;
    } else {
      return state_current;
    }
  }

  inline const PoseVelBiasState& getStateLin() const {
    return state_linearized;
  }

  inline bool isLinearized() const { return linearized; }
  inline const VecN& getDelta() const { return delta; }
  inline int64_t getT_ns() const { return state_linearized.t_ns; }

  friend struct PoseStateWithLin;

  inline void backup() {
    backup_delta = delta;
    backup_state_linearized = state_linearized;
    backup_state_current = state_current;
  }

  inline void restore() {
    delta = backup_delta;
    state_linearized = backup_state_linearized;
    state_current = backup_state_current;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  bool linearized;
  VecN delta;
  PoseVelBiasState state_linearized, state_current;

  VecN backup_delta;
  PoseVelBiasState backup_state_linearized, backup_state_current;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(state_linearized.T_w_i);
    ar(state_linearized.vel_w_i);
    ar(state_linearized.bias_gyro);
    ar(state_linearized.bias_accel);
    ar(state_current.T_w_i);
    ar(state_current.vel_w_i);
    ar(state_current.bias_gyro);
    ar(state_current.bias_accel);
    ar(delta);
    ar(linearized);
    ar(state_linearized.t_ns);
  }
};

struct PoseStateWithLin {
  using VecN = PoseState::VecN;

  PoseStateWithLin() {
    linearized = false;
    delta.setZero();
  };

  PoseStateWithLin(int64_t t_ns, const Sophus::SE3d& T_w_i,
                   bool linearized = false)
      : linearized(linearized), pose_linearized(t_ns, T_w_i) {
    delta.setZero();
    T_w_i_current = T_w_i;
  }

  PoseStateWithLin(const PoseVelBiasStateWithLin& other)
      : linearized(other.linearized),
        delta(other.delta.head<6>()),
        pose_linearized(other.state_linearized.t_ns,
                        other.state_linearized.T_w_i) {
    T_w_i_current = pose_linearized.T_w_i;
    PoseState::incPose(delta, T_w_i_current);
  }

  void setLinTrue() {
    linearized = true;
    BASALT_ASSERT(delta.isApproxToConstant(0));
    T_w_i_current = pose_linearized.T_w_i;
  }

  inline const Sophus::SE3d& getPose() const {
    if (!linearized) {
      return pose_linearized.T_w_i;
    } else {
      return T_w_i_current;
    }
  }

  inline const Sophus::SE3d& getPoseLin() const {
    return pose_linearized.T_w_i;
  }

  inline void applyInc(const VecN& inc) {
    if (!linearized) {
      PoseState::incPose(inc, pose_linearized.T_w_i);
    } else {
      delta += inc;
      T_w_i_current = pose_linearized.T_w_i;
      PoseState::incPose(delta, T_w_i_current);
    }
  }

  inline bool isLinearized() const { return linearized; }
  inline const VecN& getDelta() const { return delta; }
  inline int64_t getT_ns() const { return pose_linearized.t_ns; }

  inline void backup() {
    backup_delta = delta;
    backup_pose_linearized = pose_linearized;
    backup_T_w_i_current = T_w_i_current;
  }

  inline void restore() {
    delta = backup_delta;
    pose_linearized = backup_pose_linearized;
    T_w_i_current = backup_T_w_i_current;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  bool linearized;
  VecN delta;
  PoseState pose_linearized;
  Sophus::SE3d T_w_i_current;

  VecN backup_delta;
  PoseState backup_pose_linearized;
  Sophus::SE3d backup_T_w_i_current;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(pose_linearized.T_w_i);
    ar(T_w_i_current);
    ar(delta);
    ar(linearized);
    ar(pose_linearized.t_ns);
  }
};

struct AbsOrderMap {
  std::map<int64_t, std::pair<int, int>> abs_order_map;
  size_t items = 0;
  size_t total_size = 0;

  void print_order() {
    for (const auto& kv : abs_order_map) {
      std::cout << kv.first << " (" << kv.second.first << ","
                << kv.second.second << ")" << std::endl;
    }
    std::cout << std::endl;
  }
};

struct MargData {
  typedef std::shared_ptr<MargData> Ptr;

  AbsOrderMap aom;
  Eigen::MatrixXd abs_H;
  Eigen::VectorXd abs_b;
  Eigen::map<int64_t, PoseVelBiasStateWithLin> frame_states;
  Eigen::map<int64_t, PoseStateWithLin> frame_poses;
  std::set<int64_t> kfs_all;
  std::set<int64_t> kfs_to_marg;

  std::vector<OpticalFlowResult::Ptr> opt_flow_res;
};

struct RelPoseFactor {
  int64_t t_i_ns, t_j_ns;

  Sophus::SE3d T_i_j;
  Sophus::Matrix6d cov_inv;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RollPitchFactor {
  int64_t t_ns;

  Sophus::SO3d R_w_i_meas;

  Eigen::Matrix2d cov_inv;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace basalt

namespace cereal {

template <class Archive>
void serialize(Archive& ar, basalt::AbsOrderMap& a) {
  ar(a.total_size);
  ar(a.items);
  ar(a.abs_order_map);
}

template <class Archive>
void serialize(Archive& ar, basalt::MargData& m) {
  ar(m.aom);
  ar(m.abs_H);
  ar(m.abs_b);
  ar(m.frame_poses);
  ar(m.frame_states);
  ar(m.kfs_all);
  ar(m.kfs_to_marg);
}

}  // namespace cereal
