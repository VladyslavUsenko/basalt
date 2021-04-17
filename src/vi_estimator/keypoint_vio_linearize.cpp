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

#include <basalt/vi_estimator/keypoint_vio.h>

namespace basalt {

void KeypointVioEstimator::linearizeAbsIMU(
    const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
    double& imu_error, double& bg_error, double& ba_error,
    const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double>>& states,
    const Eigen::aligned_map<int64_t, IntegratedImuMeasurement<double>>&
        imu_meas,
    const Eigen::Vector3d& gyro_bias_weight,
    const Eigen::Vector3d& accel_bias_weight, const Eigen::Vector3d& g) {
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

      const size_t start_idx = aom.abs_order_map.at(start_t).first;
      const size_t end_idx = aom.abs_order_map.at(end_t).first;

      PoseVelBiasStateWithLin start_state = states.at(start_t);
      PoseVelBiasStateWithLin end_state = states.at(end_t);

      IntegratedImuMeasurement<double>::MatNN d_res_d_start, d_res_d_end;
      IntegratedImuMeasurement<double>::MatN3 d_res_d_bg, d_res_d_ba;

      PoseVelState<double>::VecN res = kv.second.residual(
          start_state.getStateLin(), g, end_state.getStateLin(),
          start_state.getStateLin().bias_gyro,
          start_state.getStateLin().bias_accel, &d_res_d_start, &d_res_d_end,
          &d_res_d_bg, &d_res_d_ba);

      if (start_state.isLinearized() || end_state.isLinearized()) {
        res =
            kv.second.residual(start_state.getState(), g, end_state.getState(),
                               start_state.getState().bias_gyro,
                               start_state.getState().bias_accel);
      }

      // error
      imu_error += 0.5 * res.transpose() * kv.second.get_cov_inv() * res;

      // states
      abs_H.block<9, 9>(start_idx, start_idx) +=
          d_res_d_start.transpose() * kv.second.get_cov_inv() * d_res_d_start;
      abs_H.block<9, 9>(start_idx, end_idx) +=
          d_res_d_start.transpose() * kv.second.get_cov_inv() * d_res_d_end;
      abs_H.block<9, 9>(end_idx, start_idx) +=
          d_res_d_end.transpose() * kv.second.get_cov_inv() * d_res_d_start;
      abs_H.block<9, 9>(end_idx, end_idx) +=
          d_res_d_end.transpose() * kv.second.get_cov_inv() * d_res_d_end;

      abs_b.segment<9>(start_idx) +=
          d_res_d_start.transpose() * kv.second.get_cov_inv() * res;
      abs_b.segment<9>(end_idx) +=
          d_res_d_end.transpose() * kv.second.get_cov_inv() * res;

      // bias
      IntegratedImuMeasurement<double>::MatN6 d_res_d_bga;
      d_res_d_bga.topLeftCorner<9, 3>() = d_res_d_bg;
      d_res_d_bga.topRightCorner<9, 3>() = d_res_d_ba;

      abs_H.block<6, 6>(start_idx + 9, start_idx + 9) +=
          d_res_d_bga.transpose() * kv.second.get_cov_inv() * d_res_d_bga;

      abs_H.block<9, 6>(start_idx, start_idx + 9) +=
          d_res_d_start.transpose() * kv.second.get_cov_inv() * d_res_d_bga;

      abs_H.block<9, 6>(end_idx, start_idx + 9) +=
          d_res_d_end.transpose() * kv.second.get_cov_inv() * d_res_d_bga;

      abs_H.block<6, 9>(start_idx + 9, start_idx) +=
          d_res_d_bga.transpose() * kv.second.get_cov_inv() * d_res_d_start;

      abs_H.block<6, 9>(start_idx + 9, end_idx) +=
          d_res_d_bga.transpose() * kv.second.get_cov_inv() * d_res_d_end;

      abs_b.segment<6>(start_idx + 9) +=
          d_res_d_bga.transpose() * kv.second.get_cov_inv() * res;

      // difference between biases
      double dt = kv.second.get_dt_ns() * 1e-9;
      {
        Eigen::Vector3d gyro_bias_weight_dt = gyro_bias_weight / dt;

        //        std::cerr << "gyro_bias_weight_dt " <<
        //        gyro_bias_weight_dt.transpose()
        //                  << std::endl;

        Eigen::Vector3d res_bg =
            start_state.getState().bias_gyro - end_state.getState().bias_gyro;

        abs_H.block<3, 3>(start_idx + 9, start_idx + 9) +=
            gyro_bias_weight_dt.asDiagonal();
        abs_H.block<3, 3>(end_idx + 9, end_idx + 9) +=
            gyro_bias_weight_dt.asDiagonal();

        abs_H.block<3, 3>(end_idx + 9, start_idx + 9) -=
            gyro_bias_weight_dt.asDiagonal();
        abs_H.block<3, 3>(start_idx + 9, end_idx + 9) -=
            gyro_bias_weight_dt.asDiagonal();

        abs_b.segment<3>(start_idx + 9) +=
            gyro_bias_weight_dt.asDiagonal() * res_bg;
        abs_b.segment<3>(end_idx + 9) -=
            gyro_bias_weight_dt.asDiagonal() * res_bg;

        bg_error += 0.5 * res_bg.transpose() *
                    gyro_bias_weight_dt.asDiagonal() * res_bg;
      }

      {
        Eigen::Vector3d accel_bias_weight_dt = accel_bias_weight / dt;
        Eigen::Vector3d res_ba =
            start_state.getState().bias_accel - end_state.getState().bias_accel;

        abs_H.block<3, 3>(start_idx + 12, start_idx + 12) +=
            accel_bias_weight_dt.asDiagonal();
        abs_H.block<3, 3>(end_idx + 12, end_idx + 12) +=
            accel_bias_weight_dt.asDiagonal();
        abs_H.block<3, 3>(end_idx + 12, start_idx + 12) -=
            accel_bias_weight_dt.asDiagonal();
        abs_H.block<3, 3>(start_idx + 12, end_idx + 12) -=
            accel_bias_weight_dt.asDiagonal();

        abs_b.segment<3>(start_idx + 12) +=
            accel_bias_weight_dt.asDiagonal() * res_ba;
        abs_b.segment<3>(end_idx + 12) -=
            accel_bias_weight_dt.asDiagonal() * res_ba;

        ba_error += 0.5 * res_ba.transpose() *
                    accel_bias_weight_dt.asDiagonal() * res_ba;
      }
    }
  }
}

void KeypointVioEstimator::computeImuError(
    const AbsOrderMap& aom, double& imu_error, double& bg_error,
    double& ba_error,
    const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double>>& states,
    const Eigen::aligned_map<int64_t, IntegratedImuMeasurement<double>>&
        imu_meas,
    const Eigen::Vector3d& gyro_bias_weight,
    const Eigen::Vector3d& accel_bias_weight, const Eigen::Vector3d& g) {
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

      PoseVelBiasStateWithLin start_state = states.at(start_t);
      PoseVelBiasStateWithLin end_state = states.at(end_t);

      const PoseVelState<double>::VecN res = kv.second.residual(
          start_state.getState(), g, end_state.getState(),
          start_state.getState().bias_gyro, start_state.getState().bias_accel);

      //      std::cout << "res: (" << start_t << "," << end_t << ") "
      //                << res.transpose() << std::endl;

      //      std::cerr << "cov_inv:\n" << kv.second.get_cov_inv() <<
      //      std::endl;

      imu_error += 0.5 * res.transpose() * kv.second.get_cov_inv() * res;

      double dt = kv.second.get_dt_ns() * 1e-9;
      {
        Eigen::Vector3d gyro_bias_weight_dt = gyro_bias_weight / dt;
        Eigen::Vector3d res_bg =
            start_state.getState().bias_gyro - end_state.getState().bias_gyro;

        bg_error += 0.5 * res_bg.transpose() *
                    gyro_bias_weight_dt.asDiagonal() * res_bg;
      }

      {
        Eigen::Vector3d accel_bias_weight_dt = accel_bias_weight / dt;
        Eigen::Vector3d res_ba =
            start_state.getState().bias_accel - end_state.getState().bias_accel;

        ba_error += 0.5 * res_ba.transpose() *
                    accel_bias_weight_dt.asDiagonal() * res_ba;
      }
    }
  }
}
}  // namespace basalt
