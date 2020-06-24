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
#ifndef BASALT_LINEARIZE_H
#define BASALT_LINEARIZE_H

#include <basalt/io/dataset_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/stereographic_param.hpp>

namespace basalt {

template <typename Scalar>
struct LinearizeBase {
  static const int POSE_SIZE = 6;

  static const int POS_SIZE = 3;
  static const int POS_OFFSET = 0;
  static const int ROT_SIZE = 3;
  static const int ROT_OFFSET = 3;
  static const int ACCEL_BIAS_SIZE = 9;
  static const int GYRO_BIAS_SIZE = 12;
  static const int G_SIZE = 3;

  static const int TIME_SIZE = 1;

  static const int ACCEL_BIAS_OFFSET = 0;
  static const int GYRO_BIAS_OFFSET = ACCEL_BIAS_SIZE;
  static const int G_OFFSET = GYRO_BIAS_OFFSET + GYRO_BIAS_SIZE;

  typedef Sophus::SE3<Scalar> SE3;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef typename Eigen::aligned_vector<PoseData>::const_iterator PoseDataIter;
  typedef typename Eigen::aligned_vector<GyroData>::const_iterator GyroDataIter;
  typedef
      typename Eigen::aligned_vector<AccelData>::const_iterator AccelDataIter;
  typedef typename Eigen::aligned_vector<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;

  template <int INTRINSICS_SIZE>
  struct PoseCalibH {
    Sophus::Matrix6d H_pose_accum;
    Sophus::Vector6d b_pose_accum;
    Eigen::Matrix<double, INTRINSICS_SIZE, 6> H_intr_pose_accum;
    Eigen::Matrix<double, INTRINSICS_SIZE, INTRINSICS_SIZE> H_intr_accum;
    Eigen::Matrix<double, INTRINSICS_SIZE, 1> b_intr_accum;

    PoseCalibH() {
      H_pose_accum.setZero();
      b_pose_accum.setZero();
      H_intr_pose_accum.setZero();
      H_intr_accum.setZero();
      b_intr_accum.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct CalibCommonData {
    const Calibration<Scalar>* calibration = nullptr;
    const MocapCalibration<Scalar>* mocap_calibration = nullptr;
    const Eigen::aligned_vector<Eigen::Vector4d>* aprilgrid_corner_pos_3d =
        nullptr;

    // Calib data
    const std::vector<size_t>* offset_T_i_c = nullptr;
    const std::vector<size_t>* offset_intrinsics = nullptr;

    // Cam data
    size_t mocap_block_offset;
    size_t bias_block_offset;
    const std::unordered_map<int64_t, size_t>* offset_poses = nullptr;

    // Cam-IMU data
    const Vector3* g = nullptr;

    bool opt_intrinsics;

    // Cam-IMU options
    bool opt_cam_time_offset;
    bool opt_g;
    bool opt_imu_scale;

    Scalar pose_var_inv;
    Vector3 gyro_var_inv;
    Vector3 accel_var_inv;
    Scalar mocap_var_inv;

    Scalar huber_thresh;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  template <class CamT>
  inline void linearize_point(const Eigen::Vector2d& corner_pos, int corner_id,
                              const Eigen::Matrix4d& T_c_w, const CamT& cam,
                              PoseCalibH<CamT::N>* cph, double& error,
                              int& num_points, double& reproj_err) const {
    Eigen::Matrix<double, 2, 4> d_r_d_p;
    Eigen::Matrix<double, 2, CamT::N> d_r_d_param;

    BASALT_ASSERT_STREAM(
        corner_id < int(common_data.aprilgrid_corner_pos_3d->size()),
        "corner_id " << corner_id);

    Eigen::Vector4d point3d =
        T_c_w * common_data.aprilgrid_corner_pos_3d->at(corner_id);

    Eigen::Vector2d proj;
    bool valid;
    if (cph) {
      valid = cam.project(point3d, proj, &d_r_d_p, &d_r_d_param);
    } else {
      valid = cam.project(point3d, proj);
    }
    if (!valid || !proj.array().isFinite().all()) return;

    Eigen::Vector2d residual = proj - corner_pos;

    double e = residual.norm();
    double huber_weight =
        e < common_data.huber_thresh ? 1.0 : common_data.huber_thresh / e;

    if (cph) {
      Eigen::Matrix<double, 4, 6> d_point_d_xi;

      d_point_d_xi.topLeftCorner<3, 3>().setIdentity();
      d_point_d_xi.topRightCorner<3, 3>() =
          -Sophus::SO3d::hat(point3d.head<3>());
      d_point_d_xi.row(3).setZero();

      Eigen::Matrix<double, 2, 6> d_r_d_xi = d_r_d_p * d_point_d_xi;

      cph->H_pose_accum += huber_weight * d_r_d_xi.transpose() * d_r_d_xi;
      cph->b_pose_accum += huber_weight * d_r_d_xi.transpose() * residual;

      cph->H_intr_pose_accum +=
          huber_weight * d_r_d_param.transpose() * d_r_d_xi;
      cph->H_intr_accum += huber_weight * d_r_d_param.transpose() * d_r_d_param;
      cph->b_intr_accum += huber_weight * d_r_d_param.transpose() * residual;
    }

    error += huber_weight * e * e * (2 - huber_weight);
    reproj_err += e;
    num_points++;
  }

  CalibCommonData common_data;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace basalt

#endif
