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
#include <basalt/calibration/calibration_helper.h>
#include <basalt/optimization/poses_linearize.h>

#include <tbb/parallel_reduce.h>

namespace basalt {

class PosesOptimization {
  static constexpr size_t POSE_SIZE = 6;

  using Scalar = double;

  typedef LinearizePosesOpt<Scalar, SparseHashAccumulator<Scalar>> LinearizeT;

  using SE3 = typename LinearizeT::SE3;
  using Vector2 = typename LinearizeT::Vector2;
  using Vector3 = typename LinearizeT::Vector3;
  using Vector4 = typename LinearizeT::Vector4;
  using VectorX = typename LinearizeT::VectorX;

  using AprilgridCornersDataIter =
      typename Eigen::aligned_vector<AprilgridCornersData>::const_iterator;

 public:
  PosesOptimization()
      : lambda(1e-6), min_lambda(1e-12), max_lambda(100), lambda_vee(2) {}

  Vector2 getOpticalCenter(size_t i) {
    return calib->intrinsics[i].getParam().template segment<2>(2);
  }

  void resetCalib(size_t num_cams, const std::vector<std::string> &cam_types) {
    BASALT_ASSERT(cam_types.size() == num_cams);

    calib.reset(new Calibration<Scalar>);

    for (size_t i = 0; i < cam_types.size(); i++) {
      calib->intrinsics.emplace_back(
          GenericCamera<Scalar>::fromString(cam_types[i]));

      if (calib->intrinsics.back().getName() != cam_types[i]) {
        std::cerr << "Unknown camera type " << cam_types[i] << " default to "
                  << calib->intrinsics.back().getName() << std::endl;
      }
    }
    calib->T_i_c.resize(num_cams);
  }

  void loadCalib(const std::string &p) {
    std::string path = p + "calibration.json";

    std::ifstream is(path);

    if (is.good()) {
      cereal::JSONInputArchive archive(is);
      calib.reset(new Calibration<Scalar>);
      archive(*calib);
      std::cout << "Loaded calibration from: " << path << std::endl;
    } else {
      std::cout << "No calibration found" << std::endl;
    }
  }

  void saveCalib(const std::string &path) const {
    if (calib) {
      std::ofstream os(path + "calibration.json");
      cereal::JSONOutputArchive archive(os);

      archive(*calib);
    }
  }

  bool calibInitialized() const { return calib != nullptr; }
  bool initialized() const { return true; }

  // Returns true when converged
  bool optimize(bool opt_intrinsics, double huber_thresh, double stop_thresh,
                double &error, int &num_points, double &reprojection_error) {
    error = 0;
    num_points = 0;

    ccd.opt_intrinsics = opt_intrinsics;
    ccd.huber_thresh = huber_thresh;

    LinearizePosesOpt<double, SparseHashAccumulator<double>> lopt(
        problem_size, timestam_to_pose, ccd);

    tbb::blocked_range<AprilgridCornersDataIter> april_range(
        aprilgrid_corners_measurements.begin(),
        aprilgrid_corners_measurements.end());
    tbb::parallel_reduce(april_range, lopt);

    error = lopt.error;
    num_points = lopt.num_points;
    reprojection_error = lopt.reprojection_error;

    std::cout << "[LINEARIZE] Error: " << lopt.error << " num points "
              << lopt.num_points << std::endl;

    lopt.accum.setup_solver();
    Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

    bool converged = false;
    bool step = false;
    int max_iter = 10;

    while (!step && max_iter > 0 && !converged) {
      Eigen::aligned_unordered_map<int64_t, Sophus::SE3d>
          timestam_to_pose_backup = timestam_to_pose;
      Calibration<Scalar> calib_backup = *calib;

      Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
      for (int i = 0; i < Hdiag_lambda.size(); i++)
        Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

      Eigen::VectorXd inc = -lopt.accum.solve(&Hdiag_lambda);
      double max_inc = inc.array().abs().maxCoeff();
      if (max_inc < stop_thresh) converged = true;

      for (auto &kv : timestam_to_pose) {
        kv.second *=
            Sophus::se3_expd(inc.segment<POSE_SIZE>(offset_poses[kv.first]));
      }

      for (size_t i = 0; i < calib->T_i_c.size(); i++) {
        calib->T_i_c[i] *=
            Sophus::se3_expd(inc.segment<POSE_SIZE>(offset_T_i_c[i]));
      }

      for (size_t i = 0; i < calib->intrinsics.size(); i++) {
        auto &c = calib->intrinsics[i];
        c.applyInc(inc.segment(offset_cam_intrinsics[i], c.getN()));
      }

      ComputeErrorPosesOpt<double> eopt(problem_size, timestam_to_pose, ccd);
      tbb::parallel_reduce(april_range, eopt);

      double f_diff = (lopt.error - eopt.error);
      double l_diff = 0.5 * inc.dot(inc * lambda - lopt.accum.getB());

      // std::cout << "f_diff " << f_diff << " l_diff " << l_diff << std::endl;

      double step_quality = f_diff / l_diff;

      if (step_quality < 0) {
        std::cout << "\t[REJECTED] lambda:" << lambda
                  << " step_quality: " << step_quality
                  << " max_inc: " << max_inc << " Error: " << eopt.error
                  << " num points " << eopt.num_points << std::endl;
        lambda = std::min(max_lambda, lambda_vee * lambda);
        lambda_vee *= 2;

        timestam_to_pose = timestam_to_pose_backup;
        *calib = calib_backup;
      } else {
        std::cout << "\t[ACCEPTED] lambda:" << lambda
                  << " step_quality: " << step_quality
                  << " max_inc: " << max_inc << " Error: " << eopt.error
                  << " num points " << eopt.num_points << std::endl;

        lambda = std::max(
            min_lambda,
            lambda *
                std::max(1.0 / 3, 1 - std::pow(2 * step_quality - 1, 3.0)));
        lambda_vee = 2;

        error = eopt.error;
        num_points = eopt.num_points;
        reprojection_error = eopt.reprojection_error;

        step = true;
      }
      max_iter--;
    }

    if (converged) {
      std::cout << "[CONVERGED]" << std::endl;
    }

    return converged;
  }

  void recompute_size() {
    offset_poses.clear();
    offset_T_i_c.clear();
    offset_cam_intrinsics.clear();

    size_t curr_offset = 0;

    for (const auto &kv : timestam_to_pose) {
      offset_poses[kv.first] = curr_offset;
      curr_offset += POSE_SIZE;
    }

    offset_T_i_c.emplace_back(curr_offset);
    for (size_t i = 0; i < calib->T_i_c.size(); i++)
      offset_T_i_c.emplace_back(offset_T_i_c.back() + POSE_SIZE);

    offset_cam_intrinsics.emplace_back(offset_T_i_c.back());
    for (size_t i = 0; i < calib->intrinsics.size(); i++)
      offset_cam_intrinsics.emplace_back(offset_cam_intrinsics.back() +
                                         calib->intrinsics[i].getN());

    problem_size = offset_cam_intrinsics.back();
  }

  Sophus::SE3d getT_w_i(int64_t i) {
    auto it = timestam_to_pose.find(i);

    if (it == timestam_to_pose.end())
      return Sophus::SE3d();
    else
      return it->second;
  }

  void setAprilgridCorners3d(const Eigen::aligned_vector<Eigen::Vector4d> &v) {
    aprilgrid_corner_pos_3d = v;
  }

  void init() {
    recompute_size();

    ccd.calibration = calib.get();
    ccd.aprilgrid_corner_pos_3d = &aprilgrid_corner_pos_3d;
    ccd.offset_poses = &offset_poses;
    ccd.offset_T_i_c = &offset_T_i_c;
    ccd.offset_intrinsics = &offset_cam_intrinsics;
  }

  void addAprilgridMeasurement(
      int64_t t_ns, int cam_id,
      const Eigen::aligned_vector<Eigen::Vector2d> &corners_pos,
      const std::vector<int> &corner_id) {
    aprilgrid_corners_measurements.emplace_back();

    aprilgrid_corners_measurements.back().timestamp_ns = t_ns;
    aprilgrid_corners_measurements.back().cam_id = cam_id;
    aprilgrid_corners_measurements.back().corner_pos = corners_pos;
    aprilgrid_corners_measurements.back().corner_id = corner_id;
  }

  void addPoseMeasurement(int64_t t_ns, const Sophus::SE3d &pose) {
    timestam_to_pose[t_ns] = pose;
  }

  void setVignette(const std::vector<basalt::RdSpline<1, 4>> &vign) {
    calib->vignette = vign;
  }

  void setResolution(const Eigen::aligned_vector<Eigen::Vector2i> &resolution) {
    calib->resolution = resolution;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::shared_ptr<Calibration<Scalar>> calib;

 private:
  typename LinearizePosesOpt<
      Scalar, SparseHashAccumulator<Scalar>>::CalibCommonData ccd;

  Scalar lambda, min_lambda, max_lambda, lambda_vee;

  size_t problem_size;

  std::unordered_map<int64_t, size_t> offset_poses;
  std::vector<size_t> offset_cam_intrinsics;
  std::vector<size_t> offset_T_i_c;

  // frame poses
  Eigen::aligned_unordered_map<int64_t, Sophus::SE3d> timestam_to_pose;

  Eigen::aligned_vector<AprilgridCornersData> aprilgrid_corners_measurements;

  Eigen::aligned_vector<Eigen::Vector4d> aprilgrid_corner_pos_3d;
};

}  // namespace basalt
