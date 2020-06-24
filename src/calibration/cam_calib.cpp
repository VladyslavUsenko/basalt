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

#include <basalt/calibration/cam_calib.h>

#include <basalt/utils/system_utils.h>

#include <basalt/calibration/vignette.h>

#include <basalt/optimization/poses_optimize.h>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/filesystem.h>

namespace basalt {

CamCalib::CamCalib(const std::string &dataset_path,
                   const std::string &dataset_type,
                   const std::string &aprilgrid_path,
                   const std::string &cache_path,
                   const std::string &cache_dataset_name, int skip_images,
                   const std::vector<std::string> &cam_types, bool show_gui)
    : dataset_path(dataset_path),
      dataset_type(dataset_type),
      april_grid(aprilgrid_path),
      cache_path(ensure_trailing_slash(cache_path)),
      cache_dataset_name(cache_dataset_name),
      skip_images(skip_images),
      cam_types(cam_types),
      show_gui(show_gui),
      show_frame("ui.show_frame", 0, 0, 1500),
      show_corners("ui.show_corners", true, false, true),
      show_corners_rejected("ui.show_corners_rejected", false, false, true),
      show_init_reproj("ui.show_init_reproj", false, false, true),
      show_opt("ui.show_opt", true, false, true),
      show_vign("ui.show_vign", false, false, true),
      show_ids("ui.show_ids", false, false, true),
      huber_thresh("ui.huber_thresh", 4.0, 0.1, 10.0),
      opt_intr("ui.opt_intr", true, false, true),
      opt_until_convg("ui.opt_until_converge", false, false, true),
      stop_thresh("ui.stop_thresh", 1e-8, 1e-10, 0.01, true) {
  if (show_gui) initGui();

  if (!fs::exists(cache_path)) {
    fs::create_directory(cache_path);
  }

  pangolin::ColourWheel cw;
  for (int i = 0; i < 20; i++) {
    cam_colors.emplace_back(cw.GetUniqueColour());
  }
}

CamCalib::~CamCalib() {
  if (processing_thread) {
    processing_thread->join();
  }
}

void CamCalib::initGui() {
  pangolin::CreateWindowAndBind("Main", 1600, 1000);

  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                        pangolin::Attach::Pix(UI_WIDTH));

  img_view_display =
      &pangolin::CreateDisplay()
           .SetBounds(0.5, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
           .SetLayout(pangolin::LayoutEqual);

  pangolin::View &vign_plot_display =
      pangolin::CreateDisplay().SetBounds(0.0, 0.5, 0.72, 1.0);

  vign_plotter.reset(new pangolin::Plotter(&vign_data_log, 0.0, 1000.0, 0.0,
                                           1.0, 0.01f, 0.01f));
  vign_plot_display.AddDisplay(*vign_plotter);

  pangolin::View &polar_error_display = pangolin::CreateDisplay().SetBounds(
      0.0, 0.5, pangolin::Attach::Pix(UI_WIDTH), 0.43);

  polar_plotter.reset(
      new pangolin::Plotter(nullptr, 0.0, 120.0, 0.0, 1.0, 0.01f, 0.01f));
  polar_error_display.AddDisplay(*polar_plotter);

  pangolin::View &azimuthal_plot_display =
      pangolin::CreateDisplay().SetBounds(0.0, 0.5, 0.45, 0.7);

  azimuth_plotter.reset(
      new pangolin::Plotter(nullptr, -180.0, 180.0, 0.0, 1.0, 0.01f, 0.01f));
  azimuthal_plot_display.AddDisplay(*azimuth_plotter);

  pangolin::Var<std::function<void(void)>> load_dataset(
      "ui.load_dataset", std::bind(&CamCalib::loadDataset, this));

  pangolin::Var<std::function<void(void)>> detect_corners(
      "ui.detect_corners", std::bind(&CamCalib::detectCorners, this));

  pangolin::Var<std::function<void(void)>> init_cam_intrinsics(
      "ui.init_cam_intr", std::bind(&CamCalib::initCamIntrinsics, this));

  pangolin::Var<std::function<void(void)>> init_cam_poses(
      "ui.init_cam_poses", std::bind(&CamCalib::initCamPoses, this));

  pangolin::Var<std::function<void(void)>> init_cam_extrinsics(
      "ui.init_cam_extr", std::bind(&CamCalib::initCamExtrinsics, this));

  pangolin::Var<std::function<void(void)>> init_opt(
      "ui.init_opt", std::bind(&CamCalib::initOptimization, this));

  pangolin::Var<std::function<void(void)>> optimize(
      "ui.optimize", std::bind(&CamCalib::optimize, this));

  pangolin::Var<std::function<void(void)>> save_calib(
      "ui.save_calib", std::bind(&CamCalib::saveCalib, this));

  pangolin::Var<std::function<void(void)>> compute_vign(
      "ui.compute_vign", std::bind(&CamCalib::computeVign, this));

  setNumCameras(1);
}

void CamCalib::computeVign() {
  Eigen::aligned_vector<Eigen::Vector2d> optical_centers;
  for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
    optical_centers.emplace_back(
        calib_opt->calib->intrinsics[i].getParam().segment<2>(2));
  }

  std::map<TimeCamId, Eigen::aligned_vector<Eigen::Vector3d>>
      reprojected_vignette2;
  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
    const std::vector<ImageData> img_vec =
        vio_dataset->get_image_data(timestamp_ns);

    for (size_t j = 0; j < calib_opt->calib->intrinsics.size(); j++) {
      TimeCamId tcid(timestamp_ns, j);

      auto it = reprojected_vignette.find(tcid);

      if (it != reprojected_vignette.end() && img_vec[j].img.get()) {
        Eigen::aligned_vector<Eigen::Vector3d> rv;
        rv.resize(it->second.corners_proj.size());

        for (size_t k = 0; k < it->second.corners_proj.size(); k++) {
          Eigen::Vector2d pos = it->second.corners_proj[k];

          rv[k].head<2>() = pos;

          if (img_vec[j].img->InBounds(pos[0], pos[1], 1) &&
              it->second.corners_proj_success[k]) {
            double val = img_vec[j].img->interp(pos);
            val /= std::numeric_limits<uint16_t>::max();

            if (img_vec[j].exposure > 0) {
              val *= 0.001 / img_vec[j].exposure;  // bring to common exposure
            }

            rv[k][2] = val;
          } else {
            rv[k][2] = -1;
          }
        }

        reprojected_vignette2.emplace(tcid, rv);
      }
    }
  }

  VignetteEstimator ve(vio_dataset, optical_centers,
                       calib_opt->calib->resolution, reprojected_vignette2,
                       april_grid);

  ve.optimize();
  ve.compute_error(&reprojected_vignette_error);

  std::vector<std::vector<float>> vign_data;
  ve.compute_data_log(vign_data);
  vign_data_log.Clear();
  for (const auto &v : vign_data) vign_data_log.Log(v);

  {
    vign_plotter->ClearSeries();
    vign_plotter->ClearMarkers();

    for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
      vign_plotter->AddSeries("$i", "$" + std::to_string(2 * i),
                              pangolin::DrawingModeLine, cam_colors[i],
                              "vignette camera " + std::to_string(i));
    }

    vign_plotter->ScaleViewSmooth(vign_data_log.Samples() / 1000.0f, 1.0f, 0.0f,
                                  0.5f);
  }

  ve.save_vign_png(cache_path);

  calib_opt->setVignette(ve.get_vign_param());

  std::cout << "Saved vignette png files to " << cache_path << std::endl;
}

void CamCalib::setNumCameras(size_t n) {
  while (img_view.size() < n && show_gui) {
    std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

    size_t idx = img_view.size();
    img_view.push_back(iv);

    img_view_display->AddDisplay(*iv);
    iv->extern_draw_function = std::bind(&CamCalib::drawImageOverlay, this,
                                         std::placeholders::_1, idx);
  }
}

void CamCalib::renderingLoop() {
  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (vio_dataset.get()) {
      if (show_frame.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(show_frame);
        int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

        const std::vector<ImageData> &img_vec =
            vio_dataset->get_image_data(timestamp);

        for (size_t cam_id = 0; cam_id < vio_dataset->get_num_cams(); cam_id++)
          if (img_vec[cam_id].img.get()) {
            pangolin::GlPixFormat fmt;
            fmt.glformat = GL_LUMINANCE;
            fmt.gltype = GL_UNSIGNED_SHORT;
            fmt.scalable_internal_format = GL_LUMINANCE16;

            img_view[cam_id]->SetImage(
                img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
          } else {
            img_view[cam_id]->Clear();
          }
      }
    }

    if (opt_until_convg) {
      bool converged = optimizeWithParam(true);
      if (converged) opt_until_convg = false;
    }

    pangolin::FinishFrame();
  }
}

void CamCalib::computeProjections() {
  reprojected_corners.clear();
  reprojected_vignette.clear();

  if (!calib_opt.get() || !vio_dataset.get()) return;

  constexpr int ANGLE_BIN_SIZE = 2;
  std::vector<Eigen::Matrix<double, 180 / ANGLE_BIN_SIZE, 1>> polar_sum(
      calib_opt->calib->intrinsics.size());
  std::vector<Eigen::Matrix<int, 180 / ANGLE_BIN_SIZE, 1>> polar_num(
      calib_opt->calib->intrinsics.size());

  std::vector<Eigen::Matrix<double, 360 / ANGLE_BIN_SIZE, 1>> azimuth_sum(
      calib_opt->calib->intrinsics.size());
  std::vector<Eigen::Matrix<int, 360 / ANGLE_BIN_SIZE, 1>> azimuth_num(
      calib_opt->calib->intrinsics.size());

  for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
    polar_sum[i].setZero();
    polar_num[i].setZero();
    azimuth_sum[i].setZero();
    azimuth_num[i].setZero();
  }

  for (size_t j = 0; j < vio_dataset->get_image_timestamps().size(); ++j) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[j];

    for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
      TimeCamId tcid(timestamp_ns, i);

      ProjectedCornerData rc, rv;
      Eigen::aligned_vector<Eigen::Vector2d> polar_azimuthal_angle;

      Sophus::SE3d T_c_w_ =
          (calib_opt->getT_w_i(timestamp_ns) * calib_opt->calib->T_i_c[i])
              .inverse();

      Eigen::Matrix4d T_c_w = T_c_w_.matrix();

      calib_opt->calib->intrinsics[i].project(
          april_grid.aprilgrid_corner_pos_3d, T_c_w, rc.corners_proj,
          rc.corners_proj_success, polar_azimuthal_angle);

      calib_opt->calib->intrinsics[i].project(
          april_grid.aprilgrid_vignette_pos_3d, T_c_w, rv.corners_proj,
          rv.corners_proj_success);

      reprojected_corners.emplace(tcid, rc);
      reprojected_vignette.emplace(tcid, rv);

      // Compute reprojection histogrames over polar and azimuth angle
      auto it = calib_corners.find(tcid);
      if (it != calib_corners.end()) {
        for (size_t k = 0; k < it->second.corners.size(); k++) {
          size_t id = it->second.corner_ids[k];

          if (rc.corners_proj_success[id]) {
            double error = (it->second.corners[k] - rc.corners_proj[id]).norm();

            size_t polar_bin =
                180 * polar_azimuthal_angle[id][0] / (M_PI * ANGLE_BIN_SIZE);

            polar_sum[tcid.cam_id][polar_bin] += error;
            polar_num[tcid.cam_id][polar_bin] += 1;

            size_t azimuth_bin =
                180 / ANGLE_BIN_SIZE + (180.0 * polar_azimuthal_angle[id][1]) /
                                           (M_PI * ANGLE_BIN_SIZE);

            azimuth_sum[tcid.cam_id][azimuth_bin] += error;
            azimuth_num[tcid.cam_id][azimuth_bin] += 1;
          }
        }
      }
    }
  }

  while (polar_data_log.size() < calib_opt->calib->intrinsics.size()) {
    polar_data_log.emplace_back(new pangolin::DataLog);
  }

  while (azimuth_data_log.size() < calib_opt->calib->intrinsics.size()) {
    azimuth_data_log.emplace_back(new pangolin::DataLog);
  }

  constexpr int MIN_POINTS_HIST = 3;
  polar_plotter->ClearSeries();
  azimuth_plotter->ClearSeries();

  for (size_t c = 0; c < calib_opt->calib->intrinsics.size(); c++) {
    polar_data_log[c]->Clear();
    azimuth_data_log[c]->Clear();

    for (int i = 0; i < polar_sum[c].rows(); i++) {
      if (polar_num[c][i] > MIN_POINTS_HIST) {
        double x_coord = ANGLE_BIN_SIZE * i + ANGLE_BIN_SIZE / 2.0;
        double mean_reproj = polar_sum[c][i] / polar_num[c][i];

        polar_data_log[c]->Log(x_coord, mean_reproj);
      }
    }

    polar_plotter->AddSeries(
        "$0", "$1", pangolin::DrawingModeLine, cam_colors[c],
        "mean error(pix) vs polar angle(deg) for cam" + std::to_string(c),
        polar_data_log[c].get());

    for (int i = 0; i < azimuth_sum[c].rows(); i++) {
      if (azimuth_num[c][i] > MIN_POINTS_HIST) {
        double x_coord = ANGLE_BIN_SIZE * i + ANGLE_BIN_SIZE / 2.0 - 180.0;
        double mean_reproj = azimuth_sum[c][i] / azimuth_num[c][i];

        azimuth_data_log[c]->Log(x_coord, mean_reproj);
      }
    }

    azimuth_plotter->AddSeries(
        "$0", "$1", pangolin::DrawingModeLine, cam_colors[c],
        "mean error(pix) vs azimuth angle(deg) for cam" + std::to_string(c),
        azimuth_data_log[c].get());
  }
}

void CamCalib::detectCorners() {
  if (processing_thread) {
    processing_thread->join();
    processing_thread.reset();
  }

  processing_thread.reset(new std::thread([this]() {
    std::cout << "Started detecting corners" << std::endl;

    CalibHelper::detectCorners(this->vio_dataset, this->calib_corners,
                               this->calib_corners_rejected);

    std::string path =
        cache_path + cache_dataset_name + "_detected_corners.cereal";
    std::ofstream os(path, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);

    archive(this->calib_corners);
    archive(this->calib_corners_rejected);

    std::cout << "Done detecting corners. Saved them here: " << path
              << std::endl;
  }));

  if (!show_gui) {
    processing_thread->join();
    processing_thread.reset();
  }
}

void CamCalib::initCamIntrinsics() {
  if (calib_corners.empty()) {
    std::cerr << "No corners detected. Press detect_corners to start corner "
                 "detection."
              << std::endl;
    return;
  }

  std::cout << "Started camera intrinsics initialization" << std::endl;

  if (!calib_opt) calib_opt.reset(new PosesOptimization);

  calib_opt->resetCalib(vio_dataset->get_num_cams(), cam_types);

  std::vector<bool> cam_initialized(vio_dataset->get_num_cams(), false);

  int inc = 1;
  if (vio_dataset->get_image_timestamps().size() > 100) inc = 3;

  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size();
         i += inc) {
      const int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
      const std::vector<basalt::ImageData> &img_vec =
          vio_dataset->get_image_data(timestamp_ns);

      TimeCamId tcid(timestamp_ns, j);

      if (calib_corners.find(tcid) != calib_corners.end()) {
        CalibCornerData cid = calib_corners.at(tcid);

        Eigen::Vector4d init_intr;

        bool success = CalibHelper::initializeIntrinsics(
            cid.corners, cid.corner_ids, april_grid, img_vec[j].img->w,
            img_vec[j].img->h, init_intr);

        if (success) {
          cam_initialized[j] = true;
          calib_opt->calib->intrinsics[j].setFromInit(init_intr);
          break;
        }
      }
    }
  }

  // Try perfect pinhole initialization for cameras that are not initalized.
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    if (!cam_initialized[j]) {
      std::vector<CalibCornerData *> pinhole_corners;
      int w = 0;
      int h = 0;

      for (size_t i = 0; i < vio_dataset->get_image_timestamps().size();
           i += inc) {
        const int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
        const std::vector<basalt::ImageData> &img_vec =
            vio_dataset->get_image_data(timestamp_ns);

        TimeCamId tcid(timestamp_ns, j);

        auto it = calib_corners.find(tcid);
        if (it != calib_corners.end()) {
          if (it->second.corners.size() > 8) {
            pinhole_corners.emplace_back(&it->second);
          }
        }

        w = img_vec[j].img->w;
        h = img_vec[j].img->h;
      }

      BASALT_ASSERT(w > 0 && h > 0);

      Eigen::Vector4d init_intr;

      bool success = CalibHelper::initializeIntrinsicsPinhole(
          pinhole_corners, april_grid, w, h, init_intr);

      if (success) {
        cam_initialized[j] = true;

        std::cout << "Initialized camera " << j
                  << " with pinhole model. You should set pinhole model for "
                     "this camera!"
                  << std::endl;
        calib_opt->calib->intrinsics[j].setFromInit(init_intr);
      }
    }
  }

  std::cout << "Done camera intrinsics initialization:" << std::endl;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    std::cout << "Cam " << j << ": "
              << calib_opt->calib->intrinsics[j].getParam().transpose()
              << std::endl;
  }

  // set resolution
  {
    size_t img_idx = 1;
    int64_t t_ns = vio_dataset->get_image_timestamps()[img_idx];
    auto img_data = vio_dataset->get_image_data(t_ns);

    // Find the frame with all valid images
    while (img_idx < vio_dataset->get_image_timestamps().size()) {
      bool img_data_valid = true;
      for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
        if (!img_data[i].img.get()) img_data_valid = false;
      }

      if (!img_data_valid) {
        img_idx++;
        int64_t t_ns_new = vio_dataset->get_image_timestamps()[img_idx];
        img_data = vio_dataset->get_image_data(t_ns_new);
      } else {
        break;
      }
    }

    Eigen::aligned_vector<Eigen::Vector2i> res;

    for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
      res.emplace_back(img_data[i].img->w, img_data[i].img->h);
    }

    calib_opt->setResolution(res);
  }
}

void CamCalib::initCamPoses() {
  if (calib_corners.empty()) {
    std::cerr << "No corners detected. Press detect_corners to start corner "
                 "detection."
              << std::endl;
    return;
  }

  if (!calib_opt.get() || !calib_opt->calibInitialized()) {
    std::cerr << "No initial intrinsics. Press init_intrinsics initialize "
                 "intrinsics"
              << std::endl;
    return;
  }

  if (processing_thread) {
    processing_thread->join();
    processing_thread.reset();
  }

  std::cout << "Started initial camera pose computation " << std::endl;

  CalibHelper::initCamPoses(calib_opt->calib,
                            april_grid.aprilgrid_corner_pos_3d,
                            this->calib_corners, this->calib_init_poses);

  std::string path = cache_path + cache_dataset_name + "_init_poses.cereal";
  std::ofstream os(path, std::ios::binary);
  cereal::BinaryOutputArchive archive(os);

  archive(this->calib_init_poses);

  std::cout << "Done initial camera pose computation. Saved them here: " << path
            << std::endl;
}

void CamCalib::initCamExtrinsics() {
  if (calib_init_poses.empty()) {
    std::cerr << "No initial camera poses. Press init_cam_poses initialize "
                 "camera poses "
              << std::endl;
    return;
  }

  if (!calib_opt.get() || !calib_opt->calibInitialized()) {
    std::cerr << "No initial intrinsics. Press init_intrinsics initialize "
                 "intrinsics"
              << std::endl;
    return;
  }

  // Camera graph. Stores the edge from i to j with weight w and timestamp. i
  // and j should be sorted;
  std::map<std::pair<size_t, size_t>, std::pair<int, int64_t>> cam_graph;

  // Construct the graph.
  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];

    for (size_t cam_i = 0; cam_i < vio_dataset->get_num_cams(); cam_i++) {
      TimeCamId tcid_i(timestamp_ns, cam_i);

      auto it = calib_init_poses.find(tcid_i);
      if (it == calib_init_poses.end() || it->second.num_inliers < MIN_CORNERS)
        continue;

      for (size_t cam_j = cam_i + 1; cam_j < vio_dataset->get_num_cams();
           cam_j++) {
        TimeCamId tcid_j(timestamp_ns, cam_j);

        auto it2 = calib_init_poses.find(tcid_j);
        if (it2 == calib_init_poses.end() ||
            it2->second.num_inliers < MIN_CORNERS)
          continue;

        std::pair<size_t, size_t> edge_id(cam_i, cam_j);

        int curr_weight = cam_graph[edge_id].first;
        int new_weight =
            std::min(it->second.num_inliers, it2->second.num_inliers);

        if (curr_weight < new_weight) {
          cam_graph[edge_id] = std::make_pair(new_weight, timestamp_ns);
        }
      }
    }
  }

  std::vector<bool> cameras_initialized(vio_dataset->get_num_cams(), false);
  cameras_initialized[0] = true;
  size_t last_camera = 0;
  calib_opt->calib->T_i_c[0] = Sophus::SE3d();  // Identity

  auto next_max_weight_edge = [&](size_t cam_id) {
    int max_weight = -1;
    std::pair<int, int64_t> res(-1, -1);

    for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
      if (cameras_initialized[i]) continue;

      std::pair<size_t, size_t> edge_id;

      if (i < cam_id) {
        edge_id = std::make_pair(i, cam_id);
      } else if (i > cam_id) {
        edge_id = std::make_pair(cam_id, i);
      }

      auto it = cam_graph.find(edge_id);
      if (it != cam_graph.end() && max_weight < it->second.first) {
        max_weight = it->second.first;
        res.first = i;
        res.second = it->second.second;
      }
    }

    return res;
  };

  for (size_t i = 0; i < vio_dataset->get_num_cams() - 1; i++) {
    std::pair<int, int64_t> res = next_max_weight_edge(last_camera);

    std::cout << "Initializing camera pair " << last_camera << " " << res.first
              << std::endl;

    if (res.first >= 0) {
      size_t new_camera = res.first;

      TimeCamId tcid_last(res.second, last_camera);
      TimeCamId tcid_new(res.second, new_camera);

      calib_opt->calib->T_i_c[new_camera] =
          calib_opt->calib->T_i_c[last_camera] *
          calib_init_poses.at(tcid_last).T_a_c.inverse() *
          calib_init_poses.at(tcid_new).T_a_c;

      last_camera = new_camera;
      cameras_initialized[last_camera] = true;
    }
  }

  std::cout << "Done camera extrinsics initialization:" << std::endl;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    std::cout << "T_c0_c" << j << ":\n"
              << calib_opt->calib->T_i_c[j].matrix() << std::endl;
  }
}  // namespace basalt

void CamCalib::initOptimization() {
  if (!calib_opt) {
    std::cerr << "Calibration is not initialized. Initialize calibration first!"
              << std::endl;
    return;
  }

  if (calib_init_poses.empty()) {
    std::cerr << "No initial camera poses. Press init_cam_poses initialize "
                 "camera poses "
              << std::endl;
    return;
  }

  calib_opt->setAprilgridCorners3d(april_grid.aprilgrid_corner_pos_3d);

  std::unordered_set<TimeCamId> invalid_frames;
  for (const auto &kv : calib_corners) {
    if (kv.second.corner_ids.size() < MIN_CORNERS)
      invalid_frames.insert(kv.first);
  }

  for (size_t j = 0; j < vio_dataset->get_image_timestamps().size(); ++j) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[j];

    int max_inliers = -1;
    int max_inliers_idx = -1;

    for (size_t cam_id = 0; cam_id < calib_opt->calib->T_i_c.size(); cam_id++) {
      TimeCamId tcid(timestamp_ns, cam_id);
      const auto cp_it = calib_init_poses.find(tcid);
      if (cp_it != calib_init_poses.end()) {
        if ((int)cp_it->second.num_inliers > max_inliers) {
          max_inliers = cp_it->second.num_inliers;
          max_inliers_idx = cam_id;
        }
      }
    }

    if (max_inliers >= (int)MIN_CORNERS) {
      TimeCamId tcid(timestamp_ns, max_inliers_idx);
      const auto cp_it = calib_init_poses.find(tcid);

      // Initial pose
      calib_opt->addPoseMeasurement(
          timestamp_ns, cp_it->second.T_a_c *
                            calib_opt->calib->T_i_c[max_inliers_idx].inverse());
    } else {
      // Set all frames invalid if we do not have initial pose
      for (size_t cam_id = 0; cam_id < calib_opt->calib->T_i_c.size();
           cam_id++) {
        invalid_frames.emplace(timestamp_ns, cam_id);
      }
    }
  }

  for (const auto &kv : calib_corners) {
    if (invalid_frames.count(kv.first) == 0)
      calib_opt->addAprilgridMeasurement(kv.first.frame_id, kv.first.cam_id,
                                         kv.second.corners,
                                         kv.second.corner_ids);
  }

  calib_opt->init();
  computeProjections();

  std::cout << "Initialized optimization." << std::endl;
}  // namespace basalt

void CamCalib::loadDataset() {
  basalt::DatasetIoInterfacePtr dataset_io =
      basalt::DatasetIoFactory::getDatasetIo(dataset_type);

  dataset_io->read(dataset_path);

  vio_dataset = dataset_io->get_data();
  setNumCameras(vio_dataset->get_num_cams());

  if (skip_images > 1) {
    std::vector<int64_t> new_image_timestamps;
    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
      if (i % skip_images == 0)
        new_image_timestamps.push_back(vio_dataset->get_image_timestamps()[i]);
    }

    vio_dataset->get_image_timestamps() = new_image_timestamps;
  }

  // load detected corners if they exist
  {
    std::string path =
        cache_path + cache_dataset_name + "_detected_corners.cereal";

    std::ifstream is(path, std::ios::binary);

    if (is.good()) {
      cereal::BinaryInputArchive archive(is);

      calib_corners.clear();
      calib_corners_rejected.clear();
      archive(calib_corners);
      archive(calib_corners_rejected);

      std::cout << "Loaded detected corners from: " << path << std::endl;
    } else {
      std::cout << "No pre-processed detected corners found" << std::endl;
    }
  }

  // load initial poses if they exist
  {
    std::string path = cache_path + cache_dataset_name + "_init_poses.cereal";

    std::ifstream is(path, std::ios::binary);

    if (is.good()) {
      cereal::BinaryInputArchive archive(is);

      calib_init_poses.clear();
      archive(calib_init_poses);

      std::cout << "Loaded initial poses from: " << path << std::endl;
    } else {
      std::cout << "No pre-processed initial poses found" << std::endl;
    }
  }

  // load calibration if exist
  {
    if (!calib_opt) calib_opt.reset(new PosesOptimization);

    calib_opt->loadCalib(cache_path);
  }

  reprojected_corners.clear();
  reprojected_vignette.clear();

  if (show_gui) {
    show_frame = 0;

    show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
    show_frame.Meta().gui_changed = true;
  }
}

void CamCalib::optimize() { optimizeWithParam(true); }

bool CamCalib::optimizeWithParam(bool print_info,
                                 std::map<std::string, double> *stats) {
  if (calib_init_poses.empty()) {
    std::cerr << "No initial camera poses. Press init_cam_poses initialize "
                 "camera poses "
              << std::endl;
    return true;
  }

  if (!calib_opt.get() || !calib_opt->calibInitialized()) {
    std::cerr << "No initial intrinsics. Press init_intrinsics initialize "
                 "intrinsics"
              << std::endl;
    return true;
  }

  bool converged = true;

  if (calib_opt) {
    // calib_opt->compute_projections();
    double error;
    double reprojection_error;
    int num_points;

    auto start = std::chrono::high_resolution_clock::now();

    converged = calib_opt->optimize(opt_intr, huber_thresh, stop_thresh, error,
                                    num_points, reprojection_error);

    auto finish = std::chrono::high_resolution_clock::now();

    if (stats) {
      stats->clear();

      stats->emplace("energy_error", error);
      stats->emplace("num_points", num_points);
      stats->emplace("mean_energy_error", error / num_points);
      stats->emplace("reprojection_error", reprojection_error);
      stats->emplace("mean_reprojection_error",
                     reprojection_error / num_points);
    }

    if (print_info) {
      std::cout << "==================================" << std::endl;

      for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
        std::cout << "intrinsics " << i << ": "
                  << calib_opt->calib->intrinsics[i].getParam().transpose()
                  << std::endl;
        std::cout << "T_i_c" << i << ":\n"
                  << calib_opt->calib->T_i_c[i].matrix() << std::endl;
      }

      std::cout << "Current error: " << error << " num_points " << num_points
                << " mean_error " << error / num_points
                << " reprojection_error " << reprojection_error
                << " mean reprojection " << reprojection_error / num_points
                << " opt_time "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       finish - start)
                       .count()
                << "ms." << std::endl;

      if (converged) std::cout << "Optimization Converged !!" << std::endl;

      std::cout << "==================================" << std::endl;
    }

    if (show_gui) {
      computeProjections();
    }
  }

  return converged;
}

void CamCalib::saveCalib() {
  if (calib_opt) {
    calib_opt->saveCalib(cache_path);

    std::cout << "Saved calibration in " << cache_path << "calibration.json"
              << std::endl;
  }
}

void CamCalib::drawImageOverlay(pangolin::View &v, size_t cam_id) {
  UNUSED(v);

  size_t frame_id = show_frame;

  if (vio_dataset && frame_id < vio_dataset->get_image_timestamps().size()) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[frame_id];
    TimeCamId tcid(timestamp_ns, cam_id);

    if (show_corners) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (calib_corners.find(tcid) != calib_corners.end()) {
        const CalibCornerData &cr = calib_corners.at(tcid);
        const CalibCornerData &cr_rej = calib_corners_rejected.at(tcid);

        for (size_t i = 0; i < cr.corners.size(); i++) {
          // the radius is the threshold used for maximum displacement. The
          // search region is slightly larger.
          const float radius = static_cast<float>(cr.radii[i]);
          const Eigen::Vector2f c = cr.corners[i].cast<float>();
          pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

          if (show_ids)
            pangolin::GlFont::I().Text("%d", cr.corner_ids[i]).Draw(c[0], c[1]);
        }

        pangolin::GlFont::I()
            .Text("Detected %d corners (%d rejected)", cr.corners.size(),
                  cr_rej.corners.size())
            .Draw(5, 50);

        if (show_corners_rejected) {
          glColor3f(1.0, 0.5, 0.0);

          for (size_t i = 0; i < cr_rej.corners.size(); i++) {
            // the radius is the threshold used for maximum displacement. The
            // search region is slightly larger.
            const float radius = static_cast<float>(cr_rej.radii[i]);
            const Eigen::Vector2f c = cr_rej.corners[i].cast<float>();
            pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

            if (show_ids)
              pangolin::GlFont::I()
                  .Text("%d", cr_rej.corner_ids[i])
                  .Draw(c[0], c[1]);
          }
        }

      } else {
        glLineWidth(1.0);

        pangolin::GlFont::I().Text("Corners not processed").Draw(5, 50);
      }
    }

    if (show_init_reproj) {
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (calib_init_poses.find(tcid) != calib_init_poses.end()) {
        const CalibInitPoseData &cr = calib_init_poses.at(tcid);

        for (size_t i = 0; i < cr.reprojected_corners.size(); i++) {
          Eigen::Vector2d c = cr.reprojected_corners[i];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          if (show_ids) pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
        }

        pangolin::GlFont::I()
            .Text("Initial pose with %d inliers", cr.num_inliers)
            .Draw(5, 100);

      } else {
        pangolin::GlFont::I().Text("Initial pose not processed").Draw(5, 100);
      }
    }

    if (show_opt) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 1.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (reprojected_corners.find(tcid) != reprojected_corners.end()) {
        if (calib_corners.count(tcid) > 0 &&
            calib_corners.at(tcid).corner_ids.size() >= MIN_CORNERS) {
          const auto &rc = reprojected_corners.at(tcid);

          for (size_t i = 0; i < rc.corners_proj.size(); i++) {
            if (!rc.corners_proj_success[i]) continue;

            Eigen::Vector2d c = rc.corners_proj[i];
            pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

            if (show_ids) pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        } else {
          pangolin::GlFont::I().Text("Too few corners detected.").Draw(5, 150);
        }
      }
    }

    if (show_vign) {
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (reprojected_vignette.find(tcid) != reprojected_vignette.end()) {
        if (calib_corners.count(tcid) > 0 &&
            calib_corners.at(tcid).corner_ids.size() >= MIN_CORNERS) {
          const auto &rc = reprojected_vignette.at(tcid);

          bool has_errors = false;
          auto it = reprojected_vignette_error.find(tcid);
          if (it != reprojected_vignette_error.end()) has_errors = true;

          for (size_t i = 0; i < rc.corners_proj.size(); i++) {
            if (!rc.corners_proj_success[i]) continue;

            Eigen::Vector2d c = rc.corners_proj[i].head<2>();
            pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

            if (show_ids) {
              if (has_errors) {
                pangolin::GlFont::I()
                    .Text("%d(%f)", i, it->second[i])
                    .Draw(c[0], c[1]);
              } else {
                pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
              }
            }
          }
        } else {
          pangolin::GlFont::I().Text("Too few corners detected.").Draw(5, 200);
        }
      }
    }
  }
}

bool CamCalib::hasCorners() const { return !calib_corners.empty(); }

}  // namespace basalt
