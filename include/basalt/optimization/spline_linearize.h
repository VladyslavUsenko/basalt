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
#ifndef BASALT_SPLINE_LINEARIZE_H
#define BASALT_SPLINE_LINEARIZE_H

#include <basalt/io/dataset_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/camera/generic_camera.hpp>
#include <basalt/camera/stereographic_param.hpp>

#include <basalt/optimization/linearize.h>

#include <basalt/utils/test_utils.h>

#include <tbb/blocked_range.h>
#include <tbb/tbb_stddef.h>

namespace basalt {

template <int N, typename Scalar, typename AccumT>
struct LinearizeSplineOpt : public LinearizeBase<Scalar> {
  static const int POSE_SIZE = LinearizeBase<Scalar>::POSE_SIZE;
  static const int POS_SIZE = LinearizeBase<Scalar>::POS_SIZE;
  static const int POS_OFFSET = LinearizeBase<Scalar>::POS_OFFSET;
  static const int ROT_SIZE = LinearizeBase<Scalar>::ROT_SIZE;
  static const int ROT_OFFSET = LinearizeBase<Scalar>::ROT_OFFSET;
  static const int ACCEL_BIAS_SIZE = LinearizeBase<Scalar>::ACCEL_BIAS_SIZE;
  static const int GYRO_BIAS_SIZE = LinearizeBase<Scalar>::GYRO_BIAS_SIZE;
  static const int G_SIZE = LinearizeBase<Scalar>::G_SIZE;
  static const int TIME_SIZE = LinearizeBase<Scalar>::TIME_SIZE;
  static const int ACCEL_BIAS_OFFSET = LinearizeBase<Scalar>::ACCEL_BIAS_OFFSET;
  static const int GYRO_BIAS_OFFSET = LinearizeBase<Scalar>::GYRO_BIAS_OFFSET;
  static const int G_OFFSET = LinearizeBase<Scalar>::G_OFFSET;

  typedef Sophus::SE3<Scalar> SE3;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, 2, 4> Matrix24;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef Se3Spline<N, Scalar> SplineT;

  typedef typename Eigen::aligned_deque<PoseData>::const_iterator PoseDataIter;
  typedef typename Eigen::aligned_deque<GyroData>::const_iterator GyroDataIter;
  typedef
      typename Eigen::aligned_deque<AccelData>::const_iterator AccelDataIter;
  typedef typename Eigen::aligned_deque<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;
  typedef typename Eigen::aligned_deque<MocapPoseData>::const_iterator
      MocapPoseDataIter;

  // typedef typename LinearizeBase<Scalar>::PoseCalibH PoseCalibH;
  typedef typename LinearizeBase<Scalar>::CalibCommonData CalibCommonData;

  AccumT accum;
  Scalar error;
  Scalar reprojection_error;
  int num_points;

  size_t opt_size;

  const SplineT* spline;

  LinearizeSplineOpt(size_t opt_size, const SplineT* spl,
                     const CalibCommonData& common_data)
      : opt_size(opt_size), spline(spl) {
    this->common_data = common_data;

    accum.reset(opt_size);
    error = 0;
    reprojection_error = 0;
    num_points = 0;

    BASALT_ASSERT(spline);
  }

  LinearizeSplineOpt(const LinearizeSplineOpt& other, tbb::split)
      : opt_size(other.opt_size), spline(other.spline) {
    this->common_data = other.common_data;
    accum.reset(opt_size);
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  void operator()(const tbb::blocked_range<PoseDataIter>& r) {
    for (const PoseData& pm : r) {
      int64_t start_idx;

      typename SplineT::SO3JacobianStruct J_rot;
      typename SplineT::PosJacobianStruct J_pos;

      int64_t time_ns = pm.timestamp_ns;

      // std::cout << "time " << time << std::endl;
      // std::cout << "sline.minTime() " << spline.minTime() << std::endl;

      BASALT_ASSERT_STREAM(
          time_ns >= spline->minTimeNs(),
          "time " << time_ns << " spline.minTimeNs() " << spline->minTimeNs());

      // Residual from current value of spline
      Vector3 residual_pos =
          spline->positionResidual(time_ns, pm.data.translation(), &J_pos);
      Vector3 residual_rot =
          spline->orientationResidual(time_ns, pm.data.so3(), &J_rot);

      // std::cerr << "residual_pos " << residual_pos.transpose() << std::endl;
      // std::cerr << "residual_rot " << residual_rot.transpose() << std::endl;

      BASALT_ASSERT(J_pos.start_idx == J_rot.start_idx);

      start_idx = J_pos.start_idx;

      // std::cout << "J_pos.start_idx " << J_pos.start_idx << std::endl;

      const Scalar& pose_var_inv = this->common_data.pose_var_inv;

      error += pose_var_inv *
               (residual_pos.squaredNorm() + residual_rot.squaredNorm());

      for (size_t i = 0; i < N; i++) {
        size_t start_i = (start_idx + i) * POSE_SIZE;

        // std::cout << "start_idx " << start_idx << std::endl;
        // std::cout << "start_i " << start_i << std::endl;

        BASALT_ASSERT(start_i < opt_size);

        for (size_t j = 0; j <= i; j++) {
          size_t start_j = (start_idx + j) * POSE_SIZE;

          BASALT_ASSERT(start_j < opt_size);

          accum.template addH<POS_SIZE, POS_SIZE>(
              start_i + POS_OFFSET, start_j + POS_OFFSET,
              this->common_data.pose_var_inv * J_pos.d_val_d_knot[i] *
                  J_pos.d_val_d_knot[j] * Matrix3::Identity());

          accum.template addH<ROT_SIZE, ROT_SIZE>(
              start_i + ROT_OFFSET, start_j + ROT_OFFSET,
              this->common_data.pose_var_inv *
                  J_rot.d_val_d_knot[i].transpose() * J_rot.d_val_d_knot[j]);
        }

        accum.template addB<POS_SIZE>(
            start_i + POS_OFFSET,
            pose_var_inv * J_pos.d_val_d_knot[i] * residual_pos);
        accum.template addB<ROT_SIZE>(
            start_i + ROT_OFFSET,
            pose_var_inv * J_rot.d_val_d_knot[i].transpose() * residual_rot);
      }
    }
  }

  void operator()(const tbb::blocked_range<AccelDataIter>& r) {
    // size_t num_knots = spline.numKnots();
    // size_t bias_block_offset = POSE_SIZE * num_knots;

    for (const AccelData& pm : r) {
      typename SplineT::AccelPosSO3JacobianStruct J;
      typename SplineT::Mat39 J_bias;
      Matrix3 J_g;

      int64_t t = pm.timestamp_ns;

      //      std::cout << "time " << t << std::endl;
      //      std::cout << "sline.minTime() " << spline.minTime() << std::endl;

      BASALT_ASSERT_STREAM(
          t >= spline->minTimeNs(),
          "t " << t << " spline.minTime() " << spline->minTimeNs());
      BASALT_ASSERT_STREAM(
          t <= spline->maxTimeNs(),
          "t " << t << " spline.maxTime() " << spline->maxTimeNs());

      Vector3 residual = spline->accelResidual(
          t, pm.data, this->common_data.calibration->calib_accel_bias,
          *(this->common_data.g), &J, &J_bias, &J_g);

      if (!this->common_data.opt_imu_scale) {
        J_bias.template block<3, 6>(0, 3).setZero();
      }

      //      std::cerr << "================================" << std::endl;
      //      std::cerr << "accel res: " << residual.transpose() << std::endl;
      //      std::cerr << "accel_bias.transpose(): "
      //                << this->common_data.calibration->accel_bias.transpose()
      //                << std::endl;
      //      std::cerr << "*(this->common_data.g): "
      //                << this->common_data.g->transpose() << std::endl;
      //      std::cerr << "pm.data " << pm.data.transpose() << std::endl;

      //      std::cerr << "time " << t << std::endl;
      //      std::cerr << "sline.minTime() " << spline.minTime() << std::endl;
      //      std::cerr << "sline.maxTime() " << spline.maxTime() << std::endl;
      //      std::cerr << "================================" << std::endl;

      const Vector3& accel_var_inv = this->common_data.accel_var_inv;

      error += residual.transpose() * accel_var_inv.asDiagonal() * residual;

      size_t start_bias =
          this->common_data.bias_block_offset + ACCEL_BIAS_OFFSET;
      size_t start_g = this->common_data.bias_block_offset + G_OFFSET;

      for (size_t i = 0; i < N; i++) {
        size_t start_i = (J.start_idx + i) * POSE_SIZE;

        BASALT_ASSERT(start_i < opt_size);

        for (size_t j = 0; j <= i; j++) {
          size_t start_j = (J.start_idx + j) * POSE_SIZE;

          BASALT_ASSERT(start_j < opt_size);

          accum.template addH<POSE_SIZE, POSE_SIZE>(
              start_i, start_j,
              J.d_val_d_knot[i].transpose() * accel_var_inv.asDiagonal() *
                  J.d_val_d_knot[j]);
        }
        accum.template addH<ACCEL_BIAS_SIZE, POSE_SIZE>(
            start_bias, start_i,
            J_bias.transpose() * accel_var_inv.asDiagonal() *
                J.d_val_d_knot[i]);

        if (this->common_data.opt_g) {
          accum.template addH<G_SIZE, POSE_SIZE>(
              start_g, start_i,
              J_g.transpose() * accel_var_inv.asDiagonal() * J.d_val_d_knot[i]);
        }

        accum.template addB<POSE_SIZE>(start_i, J.d_val_d_knot[i].transpose() *
                                                    accel_var_inv.asDiagonal() *
                                                    residual);
      }

      accum.template addH<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(
          start_bias, start_bias,
          J_bias.transpose() * accel_var_inv.asDiagonal() * J_bias);

      if (this->common_data.opt_g) {
        accum.template addH<G_SIZE, ACCEL_BIAS_SIZE>(
            start_g, start_bias,
            J_g.transpose() * accel_var_inv.asDiagonal() * J_bias);
        accum.template addH<G_SIZE, G_SIZE>(
            start_g, start_g,
            J_g.transpose() * accel_var_inv.asDiagonal() * J_g);
      }

      accum.template addB<ACCEL_BIAS_SIZE>(
          start_bias,
          J_bias.transpose() * accel_var_inv.asDiagonal() * residual);

      if (this->common_data.opt_g) {
        accum.template addB<G_SIZE>(
            start_g, J_g.transpose() * accel_var_inv.asDiagonal() * residual);
      }
    }
  }

  void operator()(const tbb::blocked_range<GyroDataIter>& r) {
    // size_t num_knots = spline.numKnots();
    // size_t bias_block_offset = POSE_SIZE * num_knots;

    for (const GyroData& pm : r) {
      typename SplineT::SO3JacobianStruct J;
      typename SplineT::Mat312 J_bias;

      int64_t t_ns = pm.timestamp_ns;

      BASALT_ASSERT(t_ns >= spline->minTimeNs());
      BASALT_ASSERT(t_ns <= spline->maxTimeNs());

      const Vector3& gyro_var_inv = this->common_data.gyro_var_inv;

      Vector3 residual = spline->gyroResidual(
          t_ns, pm.data, this->common_data.calibration->calib_gyro_bias, &J,
          &J_bias);

      if (!this->common_data.opt_imu_scale) {
        J_bias.template block<3, 9>(0, 3).setZero();
      }

      //      std::cerr << "==============================" << std::endl;
      //      std::cerr << "residual " << residual.transpose() << std::endl;
      //      std::cerr << "gyro_bias "
      //                << this->common_data.calibration->gyro_bias.transpose()
      //                << std::endl;
      //      std::cerr << "pm.data " << pm.data.transpose() << std::endl;
      //      std::cerr << "t_ns " << t_ns << std::endl;

      error += residual.transpose() * gyro_var_inv.asDiagonal() * residual;

      size_t start_bias =
          this->common_data.bias_block_offset + GYRO_BIAS_OFFSET;
      for (size_t i = 0; i < N; i++) {
        size_t start_i = (J.start_idx + i) * POSE_SIZE + ROT_OFFSET;

        // std::cout << "start_idx " << J.start_idx << std::endl;
        // std::cout << "start_i " << start_i << std::endl;

        BASALT_ASSERT(start_i < opt_size);

        for (size_t j = 0; j <= i; j++) {
          size_t start_j = (J.start_idx + j) * POSE_SIZE + ROT_OFFSET;

          // std::cout << "start_j " << start_j << std::endl;

          BASALT_ASSERT(start_i < opt_size);

          accum.template addH<ROT_SIZE, ROT_SIZE>(
              start_i, start_j,
              J.d_val_d_knot[i].transpose() * gyro_var_inv.asDiagonal() *
                  J.d_val_d_knot[j]);
        }
        accum.template addH<GYRO_BIAS_SIZE, ROT_SIZE>(
            start_bias, start_i,
            J_bias.transpose() * gyro_var_inv.asDiagonal() * J.d_val_d_knot[i]);
        accum.template addB<ROT_SIZE>(start_i, J.d_val_d_knot[i].transpose() *
                                                   gyro_var_inv.asDiagonal() *
                                                   residual);
      }

      accum.template addH<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(
          start_bias, start_bias,
          J_bias.transpose() * gyro_var_inv.asDiagonal() * J_bias);
      accum.template addB<GYRO_BIAS_SIZE>(
          start_bias,
          J_bias.transpose() * gyro_var_inv.asDiagonal() * residual);
    }
  }

  void operator()(const tbb::blocked_range<AprilgridCornersDataIter>& r) {
    for (const AprilgridCornersData& acd : r) {
      std::visit(
          [&](const auto& cam) {
            constexpr int INTRINSICS_SIZE =
                std::remove_reference<decltype(cam)>::type::N;

            typename SplineT::PosePosSO3JacobianStruct J;

            int64_t time_ns = acd.timestamp_ns +
                              this->common_data.calibration->cam_time_offset_ns;

            if (time_ns < spline->minTimeNs() || time_ns >= spline->maxTimeNs())
              return;

            SE3 T_w_i = spline->pose(time_ns, &J);

            Vector6 d_T_wi_d_time;
            spline->d_pose_d_t(time_ns, d_T_wi_d_time);

            SE3 T_w_c =
                T_w_i * this->common_data.calibration->T_i_c[acd.cam_id];
            SE3 T_c_w = T_w_c.inverse();
            Eigen::Matrix4d T_c_w_m = T_c_w.matrix();

            typename LinearizeBase<Scalar>::template PoseCalibH<INTRINSICS_SIZE>
                cph;

            double err = 0;
            double reproj_err = 0;
            int num_inliers = 0;

            for (size_t i = 0; i < acd.corner_pos.size(); i++) {
              this->linearize_point(acd.corner_pos[i], acd.corner_id[i],
                                    T_c_w_m, cam, &cph, err, num_inliers,
                                    reproj_err);
            }

            error += err;
            reprojection_error += reproj_err;
            num_points += num_inliers;

            Matrix6 Adj = -this->common_data.calibration->T_i_c[acd.cam_id]
                               .inverse()
                               .Adj();
            Matrix6 A_H_p_A = Adj.transpose() * cph.H_pose_accum * Adj;

            size_t T_i_c_start = this->common_data.offset_T_i_c->at(acd.cam_id);
            size_t calib_start =
                this->common_data.offset_intrinsics->at(acd.cam_id);
            size_t start_cam_time_offset =
                this->common_data.mocap_block_offset + 2 * POSE_SIZE + 1;

            for (int i = 0; i < N; i++) {
              int start_i = (J.start_idx + i) * POSE_SIZE;

              Matrix6 Ji_A_H_p_A = J.d_val_d_knot[i].transpose() * A_H_p_A;

              for (int j = 0; j <= i; j++) {
                int start_j = (J.start_idx + j) * POSE_SIZE;

                accum.template addH<POSE_SIZE, POSE_SIZE>(
                    start_i, start_j, Ji_A_H_p_A * J.d_val_d_knot[j]);
              }

              accum.template addH<POSE_SIZE, POSE_SIZE>(
                  T_i_c_start, start_i,
                  -cph.H_pose_accum * Adj * J.d_val_d_knot[i]);

              if (this->common_data.opt_intrinsics)
                accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
                    calib_start, start_i,
                    cph.H_intr_pose_accum * Adj * J.d_val_d_knot[i]);

              if (this->common_data.opt_cam_time_offset)
                accum.template addH<TIME_SIZE, POSE_SIZE>(
                    start_cam_time_offset, start_i,
                    d_T_wi_d_time.transpose() * A_H_p_A * J.d_val_d_knot[i]);

              accum.template addB<POSE_SIZE>(
                  start_i, J.d_val_d_knot[i].transpose() * Adj.transpose() *
                               cph.b_pose_accum);
            }

            accum.template addH<POSE_SIZE, POSE_SIZE>(T_i_c_start, T_i_c_start,
                                                      cph.H_pose_accum);
            accum.template addB<POSE_SIZE>(T_i_c_start, -cph.b_pose_accum);

            if (this->common_data.opt_intrinsics) {
              accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
                  calib_start, T_i_c_start, -cph.H_intr_pose_accum);
              accum.template addH<INTRINSICS_SIZE, INTRINSICS_SIZE>(
                  calib_start, calib_start, cph.H_intr_accum);
              accum.template addB<INTRINSICS_SIZE>(calib_start,
                                                   cph.b_intr_accum);
            }

            if (this->common_data.opt_cam_time_offset) {
              accum.template addH<TIME_SIZE, POSE_SIZE>(
                  start_cam_time_offset, T_i_c_start,
                  -d_T_wi_d_time.transpose() * Adj.transpose() *
                      cph.H_pose_accum);

              if (this->common_data.opt_intrinsics)
                accum.template addH<TIME_SIZE, INTRINSICS_SIZE>(
                    start_cam_time_offset, calib_start,
                    d_T_wi_d_time.transpose() * Adj.transpose() *
                        cph.H_intr_pose_accum.transpose());

              accum.template addH<TIME_SIZE, TIME_SIZE>(
                  start_cam_time_offset, start_cam_time_offset,
                  d_T_wi_d_time.transpose() * A_H_p_A * d_T_wi_d_time);

              accum.template addB<TIME_SIZE>(start_cam_time_offset,
                                             d_T_wi_d_time.transpose() *
                                                 Adj.transpose() *
                                                 cph.b_pose_accum);
            }
          },
          this->common_data.calibration->intrinsics[acd.cam_id].variant);
    }
  }

  void operator()(const tbb::blocked_range<MocapPoseDataIter>& r) {
    for (const MocapPoseData& pm : r) {
      typename SplineT::PosePosSO3JacobianStruct J_pose;

      int64_t time_ns =
          pm.timestamp_ns +
          this->common_data.mocap_calibration->mocap_time_offset_ns;

      // std::cout << "time " << time << std::endl;
      // std::cout << "sline.minTime() " << spline.minTime() << std::endl;

      if (time_ns < spline->minTimeNs() || time_ns >= spline->maxTimeNs())
        continue;

      BASALT_ASSERT_STREAM(
          time_ns >= spline->minTimeNs(),
          "time " << time_ns << " spline.minTimeNs() " << spline->minTimeNs());

      const SE3 T_moc_w = this->common_data.mocap_calibration->T_moc_w;
      const SE3 T_i_mark = this->common_data.mocap_calibration->T_i_mark;

      const SE3 T_w_i = spline->pose(time_ns, &J_pose);
      const SE3 T_moc_mark = T_moc_w * T_w_i * T_i_mark;

      const SE3 T_mark_moc_meas = pm.data.inverse();

      Vector6 residual = Sophus::se3_logd(T_mark_moc_meas * T_moc_mark);

      // TODO: check derivatives
      Matrix6 d_res_d_T_i_mark;
      Sophus::rightJacobianInvSE3Decoupled(residual, d_res_d_T_i_mark);
      Matrix6 d_res_d_T_w_i = d_res_d_T_i_mark * T_i_mark.inverse().Adj();
      Matrix6 d_res_d_T_moc_w =
          d_res_d_T_i_mark * (T_w_i * T_i_mark).inverse().Adj();

      Vector6 d_T_wi_d_time;
      spline->d_pose_d_t(time_ns, d_T_wi_d_time);

      Vector6 d_res_d_time = d_res_d_T_w_i * d_T_wi_d_time;

      size_t start_idx = J_pose.start_idx;

      size_t start_T_moc_w = this->common_data.mocap_block_offset;
      size_t start_T_i_mark = this->common_data.mocap_block_offset + POSE_SIZE;
      size_t start_mocap_time_offset =
          this->common_data.mocap_block_offset + 2 * POSE_SIZE;

      // std::cout << "J_pos.start_idx " << J_pos.start_idx << std::endl;

      const Scalar& mocap_var_inv = this->common_data.mocap_var_inv;

      error += mocap_var_inv * residual.squaredNorm();

      Matrix6 H_T_w_i = d_res_d_T_w_i.transpose() * d_res_d_T_w_i;

      for (size_t i = 0; i < N; i++) {
        size_t start_i = (start_idx + i) * POSE_SIZE;

        // std::cout << "start_idx " << start_idx << std::endl;
        // std::cout << "start_i " << start_i << std::endl;

        BASALT_ASSERT(start_i < opt_size);

        Matrix6 Ji_H_T_w_i = J_pose.d_val_d_knot[i].transpose() * H_T_w_i;

        for (size_t j = 0; j <= i; j++) {
          size_t start_j = (start_idx + j) * POSE_SIZE;

          BASALT_ASSERT(start_j < opt_size);

          accum.template addH<POSE_SIZE, POSE_SIZE>(
              start_i, start_j,
              mocap_var_inv * Ji_H_T_w_i * J_pose.d_val_d_knot[j]);
        }

        accum.template addB<POSE_SIZE>(
            start_i, mocap_var_inv * J_pose.d_val_d_knot[i].transpose() *
                         d_res_d_T_w_i.transpose() * residual);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            start_T_moc_w, start_i,
            mocap_var_inv * d_res_d_T_moc_w.transpose() * d_res_d_T_w_i *
                J_pose.d_val_d_knot[i]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            start_T_i_mark, start_i,
            mocap_var_inv * d_res_d_T_i_mark.transpose() * d_res_d_T_w_i *
                J_pose.d_val_d_knot[i]);

        accum.template addH<TIME_SIZE, POSE_SIZE>(
            start_mocap_time_offset, start_i,
            mocap_var_inv * d_res_d_time.transpose() * d_res_d_T_w_i *
                J_pose.d_val_d_knot[i]);
      }

      // start_T_moc_w
      accum.template addH<POSE_SIZE, POSE_SIZE>(
          start_T_moc_w, start_T_moc_w,
          mocap_var_inv * d_res_d_T_moc_w.transpose() * d_res_d_T_moc_w);

      // start_T_i_mark
      accum.template addH<POSE_SIZE, POSE_SIZE>(
          start_T_i_mark, start_T_moc_w,
          mocap_var_inv * d_res_d_T_i_mark.transpose() * d_res_d_T_moc_w);

      accum.template addH<POSE_SIZE, POSE_SIZE>(
          start_T_i_mark, start_T_i_mark,
          mocap_var_inv * d_res_d_T_i_mark.transpose() * d_res_d_T_i_mark);

      // start_mocap_time_offset
      accum.template addH<TIME_SIZE, POSE_SIZE>(
          start_mocap_time_offset, start_T_moc_w,
          mocap_var_inv * d_res_d_time.transpose() * d_res_d_T_moc_w);

      accum.template addH<TIME_SIZE, POSE_SIZE>(
          start_mocap_time_offset, start_T_i_mark,
          mocap_var_inv * d_res_d_time.transpose() * d_res_d_T_i_mark);

      accum.template addH<TIME_SIZE, TIME_SIZE>(
          start_mocap_time_offset, start_mocap_time_offset,
          mocap_var_inv * d_res_d_time.transpose() * d_res_d_time);

      // B
      accum.template addB<POSE_SIZE>(
          start_T_moc_w,
          mocap_var_inv * d_res_d_T_moc_w.transpose() * residual);

      accum.template addB<POSE_SIZE>(
          start_T_i_mark,
          mocap_var_inv * d_res_d_T_i_mark.transpose() * residual);

      accum.template addB<TIME_SIZE>(
          start_mocap_time_offset,
          mocap_var_inv * d_res_d_time.transpose() * residual);
    }
  }

  void join(LinearizeSplineOpt& rhs) {
    accum.join(rhs.accum);
    error += rhs.error;
    reprojection_error += rhs.reprojection_error;
    num_points += rhs.num_points;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <int N, typename Scalar>
struct ComputeErrorSplineOpt : public LinearizeBase<Scalar> {
  typedef Sophus::SE3<Scalar> SE3;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, 2, 4> Matrix24;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef Se3Spline<N, Scalar> SplineT;

  typedef typename Eigen::aligned_deque<PoseData>::const_iterator PoseDataIter;
  typedef typename Eigen::aligned_deque<GyroData>::const_iterator GyroDataIter;
  typedef
      typename Eigen::aligned_deque<AccelData>::const_iterator AccelDataIter;
  typedef typename Eigen::aligned_deque<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;
  typedef typename Eigen::aligned_deque<MocapPoseData>::const_iterator
      MocapPoseDataIter;

  // typedef typename LinearizeBase<Scalar>::PoseCalibH PoseCalibH;
  typedef typename LinearizeBase<Scalar>::CalibCommonData CalibCommonData;

  Scalar error;
  Scalar reprojection_error;
  int num_points;

  size_t opt_size;

  const SplineT* spline;

  ComputeErrorSplineOpt(size_t opt_size, const SplineT* spl,
                        const CalibCommonData& common_data)
      : opt_size(opt_size), spline(spl) {
    this->common_data = common_data;

    error = 0;
    reprojection_error = 0;
    num_points = 0;

    BASALT_ASSERT(spline);
  }

  ComputeErrorSplineOpt(const ComputeErrorSplineOpt& other, tbb::split)
      : opt_size(other.opt_size), spline(other.spline) {
    this->common_data = other.common_data;
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  void operator()(const tbb::blocked_range<PoseDataIter>& r) {
    for (const PoseData& pm : r) {
      int64_t time_ns = pm.timestamp_ns;

      BASALT_ASSERT_STREAM(
          time_ns >= spline->minTimeNs(),
          "time " << time_ns << " spline.minTimeNs() " << spline->minTimeNs());

      // Residual from current value of spline
      Vector3 residual_pos =
          spline->positionResidual(time_ns, pm.data.translation());
      Vector3 residual_rot =
          spline->orientationResidual(time_ns, pm.data.so3());

      // std::cout << "J_pos.start_idx " << J_pos.start_idx << std::endl;

      const Scalar& pose_var_inv = this->common_data.pose_var_inv;

      error += pose_var_inv *
               (residual_pos.squaredNorm() + residual_rot.squaredNorm());
    }
  }

  void operator()(const tbb::blocked_range<AccelDataIter>& r) {
    // size_t num_knots = spline.numKnots();
    // size_t bias_block_offset = POSE_SIZE * num_knots;

    for (const AccelData& pm : r) {
      int64_t t = pm.timestamp_ns;

      //      std::cout << "time " << t << std::endl;
      //      std::cout << "sline.minTime() " << spline.minTime() << std::endl;

      BASALT_ASSERT_STREAM(
          t >= spline->minTimeNs(),
          "t " << t << " spline.minTime() " << spline->minTimeNs());
      BASALT_ASSERT_STREAM(
          t <= spline->maxTimeNs(),
          "t " << t << " spline.maxTime() " << spline->maxTimeNs());

      Vector3 residual = spline->accelResidual(
          t, pm.data, this->common_data.calibration->calib_accel_bias,
          *(this->common_data.g));

      const Vector3& accel_var_inv = this->common_data.accel_var_inv;

      error += residual.transpose() * accel_var_inv.asDiagonal() * residual;
    }
  }

  void operator()(const tbb::blocked_range<GyroDataIter>& r) {
    // size_t num_knots = spline.numKnots();
    // size_t bias_block_offset = POSE_SIZE * num_knots;

    for (const GyroData& pm : r) {
      int64_t t_ns = pm.timestamp_ns;

      BASALT_ASSERT(t_ns >= spline->minTimeNs());
      BASALT_ASSERT(t_ns <= spline->maxTimeNs());

      const Vector3& gyro_var_inv = this->common_data.gyro_var_inv;

      Vector3 residual = spline->gyroResidual(
          t_ns, pm.data, this->common_data.calibration->calib_gyro_bias);

      error += residual.transpose() * gyro_var_inv.asDiagonal() * residual;
    }
  }

  void operator()(const tbb::blocked_range<AprilgridCornersDataIter>& r) {
    for (const AprilgridCornersData& acd : r) {
      std::visit(
          [&](const auto& cam) {
            int64_t time_ns = acd.timestamp_ns +
                              this->common_data.calibration->cam_time_offset_ns;

            if (time_ns < spline->minTimeNs() || time_ns >= spline->maxTimeNs())
              return;

            SE3 T_w_i = spline->pose(time_ns);
            SE3 T_w_c =
                T_w_i * this->common_data.calibration->T_i_c[acd.cam_id];
            SE3 T_c_w = T_w_c.inverse();
            Eigen::Matrix4d T_c_w_m = T_c_w.matrix();

            double err = 0;
            double reproj_err = 0;
            int num_inliers = 0;

            for (size_t i = 0; i < acd.corner_pos.size(); i++) {
              this->linearize_point(acd.corner_pos[i], acd.corner_id[i],
                                    T_c_w_m, cam, nullptr, err, num_inliers,
                                    reproj_err);
            }

            error += err;
            reprojection_error += reproj_err;
            num_points += num_inliers;
          },
          this->common_data.calibration->intrinsics[acd.cam_id].variant);
    }
  }

  void operator()(const tbb::blocked_range<MocapPoseDataIter>& r) {
    for (const MocapPoseData& pm : r) {
      int64_t time_ns =
          pm.timestamp_ns +
          this->common_data.mocap_calibration->mocap_time_offset_ns;

      if (time_ns < spline->minTimeNs() || time_ns >= spline->maxTimeNs())
        continue;

      BASALT_ASSERT_STREAM(
          time_ns >= spline->minTimeNs(),
          "time " << time_ns << " spline.minTimeNs() " << spline->minTimeNs());

      const SE3 T_moc_w = this->common_data.mocap_calibration->T_moc_w;
      const SE3 T_i_mark = this->common_data.mocap_calibration->T_i_mark;

      const SE3 T_w_i = spline->pose(time_ns);
      const SE3 T_moc_mark = T_moc_w * T_w_i * T_i_mark;

      const SE3 T_mark_moc_meas = pm.data.inverse();

      Vector6 residual = Sophus::se3_logd(T_mark_moc_meas * T_moc_mark);

      const Scalar& mocap_var_inv = this->common_data.mocap_var_inv;

      error += mocap_var_inv * residual.squaredNorm();
    }
  }

  void join(ComputeErrorSplineOpt& rhs) {
    error += rhs.error;
    reprojection_error += rhs.reprojection_error;
    num_points += rhs.num_points;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace basalt

#endif
