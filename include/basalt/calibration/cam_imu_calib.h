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

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <Eigen/Dense>

#include <iostream>
#include <limits>
#include <thread>

#include <basalt/calibration/aprilgrid.h>
#include <basalt/calibration/calibration_helper.h>
#include <basalt/utils/test_utils.h>
#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

template <int N, typename Scalar>
class SplineOptimization;

class CamImuCalib {
 public:
  CamImuCalib(const std::string &dataset_path, const std::string &dataset_type,
              const std::string &aprilgrid_path, const std::string &cache_path,
              const std::string &cache_dataset_name, int skip_images,
              const std::vector<double> &imu_noise, bool show_gui = true);

  ~CamImuCalib();

  void initGui();

  void setNumCameras(size_t n);

  void renderingLoop();

  void computeProjections();

  void detectCorners();

  void initCamPoses();

  void initCamImuTransform();

  void initOptimization();

  void initMocap();

  void loadDataset();

  void optimize();

  bool optimizeWithParam(bool print_info,
                         std::map<std::string, double> *stats = nullptr);

  void saveCalib();

  void saveMocapCalib();

  void drawImageOverlay(pangolin::View &v, size_t cam_id);

  void recomputeDataLog();

  void drawPlots();

  bool hasCorners() const;

  void setOptIntrinsics(bool opt) { opt_intr = opt; }

 private:
  static constexpr int UI_WIDTH = 300;

  VioDatasetPtr vio_dataset;

  CalibCornerMap calib_corners;
  CalibCornerMap calib_corners_rejected;
  CalibInitPoseMap calib_init_poses;

  std::shared_ptr<std::thread> processing_thread;

  std::shared_ptr<SplineOptimization<5, double>> calib_opt;

  std::map<TimeCamId, ProjectedCornerData> reprojected_corners;

  std::string dataset_path;
  std::string dataset_type;

  AprilGrid april_grid;

  std::string cache_path;
  std::string cache_dataset_name;

  int skip_images;

  bool show_gui;

  const size_t MIN_CORNERS = 15;

  std::vector<double> imu_noise;

  //////////////////////

  pangolin::Var<int> show_frame;

  pangolin::Var<bool> show_corners;
  pangolin::Var<bool> show_corners_rejected;
  pangolin::Var<bool> show_init_reproj;
  pangolin::Var<bool> show_opt;
  pangolin::Var<bool> show_ids;

  pangolin::Var<bool> show_accel;
  pangolin::Var<bool> show_gyro;
  pangolin::Var<bool> show_pos;
  pangolin::Var<bool> show_rot_error;

  pangolin::Var<bool> show_mocap;
  pangolin::Var<bool> show_mocap_rot_error;
  pangolin::Var<bool> show_mocap_rot_vel;

  pangolin::Var<bool> show_spline;
  pangolin::Var<bool> show_data;

  pangolin::Var<bool> opt_intr;
  pangolin::Var<bool> opt_poses;
  pangolin::Var<bool> opt_corners;
  pangolin::Var<bool> opt_cam_time_offset;
  pangolin::Var<bool> opt_imu_scale;

  pangolin::Var<bool> opt_mocap;

  pangolin::Var<double> huber_thresh;

  pangolin::Var<bool> opt_until_convg;
  pangolin::Var<double> stop_thresh;

  pangolin::Plotter *plotter;
  pangolin::View *img_view_display;

  std::vector<std::shared_ptr<pangolin::ImageView>> img_view;

  pangolin::DataLog imu_data_log, pose_data_log, mocap_data_log, vign_data_log;
};

}  // namespace basalt
