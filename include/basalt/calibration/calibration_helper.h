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

#include <basalt/calibration/aprilgrid.h>
#include <basalt/io/dataset_io.h>
#include <basalt/utils/common_types.h>
#include <basalt/calibration/calibration.hpp>

#include <tbb/concurrent_unordered_map.h>

namespace basalt {

struct CalibCornerData {
  Eigen::aligned_vector<Eigen::Vector2d> corners;
  std::vector<int> corner_ids;
  std::vector<double> radii;  //!< threshold used for maximum displacement
                              //! during sub-pix refinement; Search region is
  //! slightly larger.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ProjectedCornerData {
  Eigen::aligned_vector<Eigen::Vector2d> corners_proj;
  std::vector<bool> corners_proj_success;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CalibInitPoseData {
  Sophus::SE3d T_a_c;
  size_t num_inliers;

  Eigen::aligned_vector<Eigen::Vector2d> reprojected_corners;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using CalibCornerMap = tbb::concurrent_unordered_map<TimeCamId, CalibCornerData,
                                                     std::hash<TimeCamId>>;

using CalibInitPoseMap =
    tbb::concurrent_unordered_map<TimeCamId, CalibInitPoseData,
                                  std::hash<TimeCamId>>;

class CalibHelper {
 public:
  static void detectCorners(const VioDatasetPtr& vio_data,
                            CalibCornerMap& calib_corners,
                            CalibCornerMap& calib_corners_rejected);

  static void initCamPoses(
      const Calibration<double>::Ptr& calib,
      const Eigen::aligned_vector<Eigen::Vector4d>& aprilgrid_corner_pos_3d,
      CalibCornerMap& calib_corners, CalibInitPoseMap& calib_init_poses);

  static bool initializeIntrinsics(
      const Eigen::aligned_vector<Eigen::Vector2d>& corners,
      const std::vector<int>& corner_ids, const AprilGrid& aprilgrid, int cols,
      int rows, Eigen::Vector4d& init_intr);

  static bool initializeIntrinsicsPinhole(
      const std::vector<CalibCornerData*> pinhole_corners,
      const AprilGrid& aprilgrid, int cols, int rows,
      Eigen::Vector4d& init_intr);

 private:
  inline static double square(double x) { return x * x; }

  inline static double hypot(double a, double b) {
    return sqrt(square(a) + square(b));
  }

  static void computeInitialPose(
      const Calibration<double>::Ptr& calib, size_t cam_id,
      const Eigen::aligned_vector<Eigen::Vector4d>& aprilgrid_corner_pos_3d,
      const basalt::CalibCornerData& cd, basalt::CalibInitPoseData& cp);

  static size_t computeReprojectionError(
      const UnifiedCamera<double>& cam_calib,
      const Eigen::aligned_vector<Eigen::Vector2d>& corners,
      const std::vector<int>& corner_ids,
      const Eigen::aligned_vector<Eigen::Vector4d>& aprilgrid_corner_pos_3d,
      const Sophus::SE3d& T_target_camera, double& error);
};

}  // namespace basalt

namespace cereal {
template <class Archive>
void serialize(Archive& ar, basalt::CalibCornerData& c) {
  ar(c.corners, c.corner_ids, c.radii);
}

template <class Archive>
void serialize(Archive& ar, basalt::CalibInitPoseData& c) {
  ar(c.T_a_c, c.num_inliers, c.reprojected_corners);
}
}  // namespace cereal
