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

#include <basalt/calibration/vignette.h>

#include <opencv2/highgui/highgui.hpp>

namespace basalt {

VignetteEstimator::VignetteEstimator(
    const VioDatasetPtr &vio_dataset,
    const Eigen::aligned_vector<Eigen::Vector2d> &optical_centers,
    const Eigen::aligned_vector<Eigen::Vector2i> &resolutions,
    const std::map<TimeCamId, Eigen::aligned_vector<Eigen::Vector3d>>
        &reprojected_vignette,
    const AprilGrid &april_grid)
    : vio_dataset(vio_dataset),
      optical_centers(optical_centers),
      resolutions(resolutions),
      reprojected_vignette(reprojected_vignette),
      april_grid(april_grid),
      vign_param(vio_dataset->get_num_cams(),
                 RdSpline<1, SPLINE_N>(knot_spacing)) {
  vign_size = 0;

  for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
    Eigen::Vector2d oc = optical_centers[i];

    size_t new_size = oc.norm() * 1.1;
    vign_size = std::max(vign_size, new_size);
  }

  // std::cerr << vign_size << std::endl;

  for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
    while (vign_param[i].maxTimeNs() <
           int64_t(vign_size) * int64_t(1e9 * 0.7)) {
      vign_param[i].knotsPushBack(Eigen::Matrix<double, 1, 1>(1));
    }

    while (vign_param[i].maxTimeNs() < int64_t(vign_size) * int64_t(1e9)) {
      vign_param[i].knotsPushBack(Eigen::Matrix<double, 1, 1>(0.01));
    }
  }

  irradiance.resize(april_grid.aprilgrid_vignette_pos_3d.size());
  std::fill(irradiance.begin(), irradiance.end(), 1.0);
}

void VignetteEstimator::compute_error(
    std::map<TimeCamId, std::vector<double>> *reprojected_vignette_error) {
  double error = 0;
  double mean_residual = 0;
  double max_residual = 0;
  int num_residuals = 0;

  TimeCamId tcid_max;
  // int point_id = 0;

  if (reprojected_vignette_error) reprojected_vignette_error->clear();

  for (const auto &kv : reprojected_vignette) {
    const TimeCamId &tcid = kv.first;
    const auto &points_2d_val = kv.second;

    Eigen::Vector2d oc = optical_centers[tcid.cam_id];

    BASALT_ASSERT(points_2d_val.size() ==
                  april_grid.aprilgrid_vignette_pos_3d.size());

    std::vector<double> ve(april_grid.aprilgrid_vignette_pos_3d.size());

    for (size_t i = 0; i < points_2d_val.size(); i++) {
      if (points_2d_val[i][2] >= 0) {
        double val = points_2d_val[i][2];
        int64_t loc =
            (points_2d_val[i].head<2>() - oc).norm() * 1e9;  // in pixels * 1e9
        double e =
            irradiance[i] * vign_param[tcid.cam_id].evaluate(loc)[0] - val;
        ve[i] = e;
        error += e * e;
        mean_residual += std::abs(e);
        max_residual = std::max(max_residual, std::abs(e));
        if (max_residual == std::abs(e)) {
          tcid_max = tcid;
          // point_id = i;
        }
        num_residuals++;
      }
    }

    if (reprojected_vignette_error)
      reprojected_vignette_error->emplace(tcid, ve);
  }

  //  std::cerr << "error " << error << std::endl;
  //  std::cerr << "mean_residual " << mean_residual / num_residuals <<
  //  std::endl;
  //  std::cerr << "max_residual " << max_residual << std::endl;

  // int frame_id = 0;
  //  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
  //    if (tcid_max.first == vio_dataset->get_image_timestamps()[i]) {
  //      frame_id = i;
  //    }
  //  }

  //  std::cerr << "tcid_max " << frame_id << " " << tcid_max.second << " point
  //  id "
  //            << point_id << std::endl
  //            << std::endl;
}

void VignetteEstimator::opt_irradience() {
  std::vector<double> new_irradiance(irradiance.size(), 0);
  std::vector<int> new_irradiance_count(irradiance.size(), 0);

  for (const auto &kv : reprojected_vignette) {
    const TimeCamId &tcid = kv.first;
    const auto &points_2d_val = kv.second;

    Eigen::Vector2d oc = optical_centers[tcid.cam_id];

    BASALT_ASSERT(points_2d_val.size() ==
                  april_grid.aprilgrid_vignette_pos_3d.size());

    for (size_t i = 0; i < points_2d_val.size(); i++) {
      if (points_2d_val[i][2] >= 0) {
        double val = points_2d_val[i][2];
        int64_t loc =
            (points_2d_val[i].head<2>() - oc).norm() * 1e9;  // in pixels * 1e9

        new_irradiance[i] += val / vign_param[tcid.cam_id].evaluate(loc)[0];
        new_irradiance_count[i] += 1;
      }
    }
  }

  for (size_t i = 0; i < irradiance.size(); i++) {
    if (new_irradiance_count[i] > 0)
      irradiance[i] = new_irradiance[i] / new_irradiance_count[i];
  }
}

void VignetteEstimator::opt_vign() {
  size_t num_knots = vign_param[0].getKnots().size();

  std::vector<std::vector<double>> new_vign_param(
      vio_dataset->get_num_cams(), std::vector<double>(num_knots, 0));
  std::vector<std::vector<double>> new_vign_param_count(
      vio_dataset->get_num_cams(), std::vector<double>(num_knots, 0));

  for (const auto &kv : reprojected_vignette) {
    const TimeCamId &tcid = kv.first;
    const auto &points_2d_val = kv.second;

    //      Sophus::SE3d T_w_cam =
    //          calib_opt->getT_w_i(tcid.first) *
    //          calib_opt->getCamT_i_c(tcid.second);
    //      Eigen::Vector3d opt_axis_w = T_w_cam.so3() *
    //      Eigen::Vector3d::UnitZ();
    //      if (-opt_axis_w[2] < angle_threshold) continue;

    Eigen::Vector2d oc = optical_centers[tcid.cam_id];

    BASALT_ASSERT(points_2d_val.size() ==
                  april_grid.aprilgrid_vignette_pos_3d.size());

    for (size_t i = 0; i < points_2d_val.size(); i++) {
      if (points_2d_val[i][2] >= 0) {
        double val = points_2d_val[i][2];
        int64_t loc = (points_2d_val[i].head<2>() - oc).norm() * 1e9;

        RdSpline<1, SPLINE_N>::JacobianStruct J;
        vign_param[tcid.cam_id].evaluate(loc, &J);

        for (size_t k = 0; k < J.d_val_d_knot.size(); k++) {
          new_vign_param[tcid.cam_id][J.start_idx + k] +=
              J.d_val_d_knot[k] * val / irradiance[i];
          new_vign_param_count[tcid.cam_id][J.start_idx + k] +=
              J.d_val_d_knot[k];
        }
      }
    }
  }

  double max_val = 0;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++)
    for (size_t i = 0; i < num_knots; i++) {
      if (new_vign_param_count[j][i] > 0) {
        // std::cerr << "update " << i << " " << j << std::endl;
        double val = new_vign_param[j][i] / new_vign_param_count[j][i];
        max_val = std::max(max_val, val);
        vign_param[j].getKnot(i)[0] = val;
      }
    }

  // normalize vignette
  double max_val_inv = 1.0 / max_val;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++)
    for (size_t i = 0; i < num_knots; i++) {
      if (new_vign_param_count[j][i] > 0) {
        vign_param[j].getKnot(i)[0] *= max_val_inv;
      }
    }
}

void VignetteEstimator::optimize() {
  compute_error();
  for (int i = 0; i < 10; i++) {
    opt_irradience();
    compute_error();
    opt_vign();
    compute_error();
  }
}

void VignetteEstimator::compute_data_log(
    std::vector<std::vector<float>> &vign_data_log) {
  std::vector<std::vector<double>> num_proj_points(
      vio_dataset->get_num_cams(), std::vector<double>(vign_size, 0));

  for (const auto &kv : reprojected_vignette) {
    const TimeCamId &tcid = kv.first;
    const auto &points_2d = kv.second;

    Eigen::Vector2d oc = optical_centers[tcid.cam_id];

    BASALT_ASSERT(points_2d.size() ==
                  april_grid.aprilgrid_vignette_pos_3d.size());

    for (size_t i = 0; i < points_2d.size(); i++) {
      if (points_2d[i][2] >= 0) {
        size_t loc = (points_2d[i].head<2>() - oc).norm();

        if (loc < vign_size) num_proj_points[tcid.cam_id][loc] += 1.;
      }
    }
  }

  vign_data_log.clear();
  for (size_t i = 0; i < vign_size; i++) {
    std::vector<float> log_data;
    for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
      int64_t loc = i * 1e9;
      log_data.push_back(vign_param[j].evaluate(loc)[0]);
      log_data.push_back(num_proj_points[j][i]);
    }
    vign_data_log.push_back(log_data);
  }
}

void VignetteEstimator::save_vign_png(const std::string &path) {
  for (size_t k = 0; k < vio_dataset->get_num_cams(); k++) {
    ManagedImage<uint16_t> vign_img(resolutions[k][0], resolutions[k][1]);
    vign_img.Fill(0);

    Eigen::Vector2d oc = optical_centers[k];

    for (size_t x = 0; x < vign_img.w; x++) {
      for (size_t y = 0; y < vign_img.h; y++) {
        int64_t loc = (Eigen::Vector2d(x, y) - oc).norm() * 1e9;
        double val = vign_param[k].evaluate(loc)[0];
        if (val < 0.5) continue;
        uint16_t val_int =
            val >= 1.0 ? std::numeric_limits<uint16_t>::max()
                       : uint16_t(val * std::numeric_limits<uint16_t>::max());
        vign_img(x, y) = val_int;
      }
    }

    //    pangolin::SaveImage(vign_img.UnsafeReinterpret<uint8_t>(),
    //                        pangolin::PixelFormatFromString("GRAY16LE"),
    //                        path + "/vingette_" + std::to_string(k) + ".png");

    cv::Mat img(vign_img.h, vign_img.w, CV_16U, vign_img.ptr);
    cv::imwrite(path + "/vingette_" + std::to_string(k) + ".png", img);
  }
}
}  // namespace basalt
