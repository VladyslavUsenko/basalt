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
#ifndef BASALT_POSES_LINEARIZE_H
#define BASALT_POSES_LINEARIZE_H

#include <basalt/io/dataset_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/optimization/linearize.h>

#include <basalt/optimization/accumulator.h>

#include <tbb/blocked_range.h>

namespace basalt {

template <typename Scalar, typename AccumT>
struct LinearizePosesOpt : public LinearizeBase<Scalar> {
  static const int POSE_SIZE = LinearizeBase<Scalar>::POSE_SIZE;

  typedef Sophus::SE3<Scalar> SE3;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef typename Eigen::aligned_vector<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;

  typedef typename LinearizeBase<Scalar>::CalibCommonData CalibCommonData;

  AccumT accum;
  Scalar error;
  Scalar reprojection_error;
  int num_points;

  size_t opt_size;

  const Eigen::aligned_unordered_map<int64_t, SE3>& timestam_to_pose;

  LinearizePosesOpt(
      size_t opt_size,
      const Eigen::aligned_unordered_map<int64_t, SE3>& timestam_to_pose,
      const CalibCommonData& common_data)
      : opt_size(opt_size), timestam_to_pose(timestam_to_pose) {
    this->common_data = common_data;
    accum.reset(opt_size);
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }
  LinearizePosesOpt(const LinearizePosesOpt& other, tbb::split)
      : opt_size(other.opt_size), timestam_to_pose(other.timestam_to_pose) {
    this->common_data = other.common_data;
    accum.reset(opt_size);
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  void operator()(const tbb::blocked_range<AprilgridCornersDataIter>& r) {
    for (const AprilgridCornersData& acd : r) {
      std::visit(
          [&](const auto& cam) {
            constexpr int INTRINSICS_SIZE =
                std::remove_reference<decltype(cam)>::type::N;
            typename LinearizeBase<Scalar>::template PoseCalibH<INTRINSICS_SIZE>
                cph;

            SE3 T_w_i = timestam_to_pose.at(acd.timestamp_ns);
            SE3 T_w_c =
                T_w_i * this->common_data.calibration->T_i_c[acd.cam_id];
            SE3 T_c_w = T_w_c.inverse();
            Eigen::Matrix4d T_c_w_m = T_c_w.matrix();

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

            const Matrix6 Adj =
                -this->common_data.calibration->T_i_c[acd.cam_id]
                     .inverse()
                     .Adj();

            const size_t po =
                this->common_data.offset_poses->at(acd.timestamp_ns);
            const size_t co = this->common_data.offset_T_i_c->at(acd.cam_id);
            const size_t io =
                this->common_data.offset_intrinsics->at(acd.cam_id);

            accum.template addH<POSE_SIZE, POSE_SIZE>(
                po, po, Adj.transpose() * cph.H_pose_accum * Adj);
            accum.template addB<POSE_SIZE>(po,
                                           Adj.transpose() * cph.b_pose_accum);

            if (acd.cam_id > 0) {
              accum.template addH<POSE_SIZE, POSE_SIZE>(
                  co, po, -cph.H_pose_accum * Adj);
              accum.template addH<POSE_SIZE, POSE_SIZE>(co, co,
                                                        cph.H_pose_accum);

              accum.template addB<POSE_SIZE>(co, -cph.b_pose_accum);
            }

            if (this->common_data.opt_intrinsics) {
              accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
                  io, po, cph.H_intr_pose_accum * Adj);

              if (acd.cam_id > 0)
                accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
                    io, co, -cph.H_intr_pose_accum);

              accum.template addH<INTRINSICS_SIZE, INTRINSICS_SIZE>(
                  io, io, cph.H_intr_accum);
              accum.template addB<INTRINSICS_SIZE>(io, cph.b_intr_accum);
            }
          },
          this->common_data.calibration->intrinsics[acd.cam_id].variant);
    }
  }

  void join(LinearizePosesOpt& rhs) {
    accum.join(rhs.accum);
    error += rhs.error;
    reprojection_error += rhs.reprojection_error;
    num_points += rhs.num_points;
  }
};

template <typename Scalar>
struct ComputeErrorPosesOpt : public LinearizeBase<Scalar> {
  static const int POSE_SIZE = LinearizeBase<Scalar>::POSE_SIZE;

  typedef Sophus::SE3<Scalar> SE3;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef typename Eigen::aligned_vector<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;

  typedef typename LinearizeBase<Scalar>::CalibCommonData CalibCommonData;

  Scalar error;
  Scalar reprojection_error;
  int num_points;

  size_t opt_size;

  const Eigen::aligned_unordered_map<int64_t, SE3>& timestam_to_pose;

  ComputeErrorPosesOpt(
      size_t opt_size,
      const Eigen::aligned_unordered_map<int64_t, SE3>& timestam_to_pose,
      const CalibCommonData& common_data)
      : opt_size(opt_size), timestam_to_pose(timestam_to_pose) {
    this->common_data = common_data;
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  ComputeErrorPosesOpt(const ComputeErrorPosesOpt& other, tbb::split)
      : opt_size(other.opt_size), timestam_to_pose(other.timestam_to_pose) {
    this->common_data = other.common_data;
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  void operator()(const tbb::blocked_range<AprilgridCornersDataIter>& r) {
    for (const AprilgridCornersData& acd : r) {
      std::visit(
          [&](const auto& cam) {
            SE3 T_w_i = timestam_to_pose.at(acd.timestamp_ns);
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

  void join(ComputeErrorPosesOpt& rhs) {
    error += rhs.error;
    reprojection_error += rhs.reprojection_error;
    num_points += rhs.num_points;
  }
};  // namespace basalt

}  // namespace basalt

#endif
