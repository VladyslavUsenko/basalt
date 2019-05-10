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

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace basalt {

CamCalib::CamCalib(const std::string &dataset_path,
                   const std::string &dataset_type,
                   const std::string &cache_path,
                   const std::string &cache_dataset_name, int skip_images,
                   const std::vector<std::string> &cam_types, bool show_gui)
    : dataset_path(dataset_path),
      dataset_type(dataset_type),
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
      opt_intr("ui.opt_intr", true, false, true) {
  if (show_gui) initGui();

  if (!fs::exists(cache_path)) {
    fs::create_directory(cache_path);
  }
}

CamCalib::~CamCalib() {
  if (processing_thread) {
    processing_thread->join();
  }
}

void CamCalib::initGui() {
  pangolin::CreateWindowAndBind("Main", 1600, 1000);

  img_view_display =
      &pangolin::CreateDisplay()
           .SetBounds(0.5, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
           .SetLayout(pangolin::LayoutEqual);

  pangolin::View &vign_plot_display =
      pangolin::CreateDisplay().SetBounds(0.0, 0.5, 0.7, 1.0);

  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                        pangolin::Attach::Pix(UI_WIDTH));

  vign_plotter.reset(new pangolin::Plotter(&vign_data_log, 0.0, 1000.0, 0.0,
                                           1.0, 0.01f, 0.01f));
  vign_plot_display.AddDisplay(*vign_plotter);

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
  Eigen::vector<Eigen::Vector2d> optical_centers;
  for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
    optical_centers.emplace_back(
        calib_opt->calib->intrinsics[i].getParam().segment<2>(2));
  }

  std::map<TimeCamId, Eigen::vector<Eigen::Vector3d>> reprojected_vignette2;
  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
    const std::vector<ImageData> img_vec =
        vio_dataset->get_image_data(timestamp_ns);

    for (size_t j = 0; j < calib_opt->calib->intrinsics.size(); j++) {
      TimeCamId tcid(timestamp_ns, j);

      auto it = reprojected_vignette.find(tcid);

      if (it != reprojected_vignette.end() && img_vec[j].img.get()) {
        Eigen::vector<Eigen::Vector3d> rv;
        rv.resize(it->second.size());

        for (size_t k = 0; k < it->second.size(); k++) {
          Eigen::Vector2d pos = it->second[k];

          rv[k].head<2>() = pos;

          if (img_vec[j].img->InBounds(pos[0], pos[1], 1)) {
            double val = img_vec[j].img->interp(pos);
            val /= std::numeric_limits<uint16_t>::max();
            rv[k][2] = val;
          } else {
            // invalid projection
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
  ve.compute_data_log(vign_data_log);

  {
    vign_plotter->ClearSeries();
    vign_plotter->ClearMarkers();

    for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
      vign_plotter->AddSeries("$i", "$" + std::to_string(2 * i),
                              pangolin::DrawingModeLine,
                              pangolin::Colour::Unspecified(),
                              "vignette camera " + std::to_string(i));
    }

    vign_plotter->ScaleViewSmooth(vign_data_log.Samples() / 1000.0f, 1.0f, 0.0f,
                                  0.5f);
  }

  ve.save_vign_png(cache_path);

  calib_opt->setVignette(ve.get_vign_param());
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

    pangolin::FinishFrame();
  }
}

void CamCalib::computeProjections() {
  reprojected_corners.clear();
  reprojected_vignette.clear();

  if (!calib_opt.get() || !vio_dataset.get()) return;

  for (size_t j = 0; j < vio_dataset->get_image_timestamps().size(); ++j) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[j];

    for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
      TimeCamId tcid = std::make_pair(timestamp_ns, i);

      Eigen::vector<Eigen::Vector2d> rc, rv;

      Sophus::SE3d T_c_w_ =
          (calib_opt->getT_w_i(timestamp_ns) * calib_opt->calib->T_i_c[i])
              .inverse();

      Eigen::Matrix4d T_c_w = T_c_w_.matrix();

      rc.resize(april_grid.aprilgrid_corner_pos_3d.size());
      rv.resize(april_grid.aprilgrid_vignette_pos_3d.size());

      std::vector<bool> rc_success, rv_success;

      calib_opt->calib->intrinsics[i].project(
          april_grid.aprilgrid_corner_pos_3d, T_c_w, rc, rc_success);

      calib_opt->calib->intrinsics[i].project(
          april_grid.aprilgrid_vignette_pos_3d, T_c_w, rv, rv_success);

      reprojected_corners.emplace(tcid, rc);
      reprojected_vignette.emplace(tcid, rv);
    }
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

  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
      const int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
      const std::vector<basalt::ImageData> &img_vec =
          vio_dataset->get_image_data(timestamp_ns);

      TimeCamId tcid = std::make_pair(timestamp_ns, j);

      if (calib_corners.find(tcid) != calib_corners.end()) {
        CalibCornerData cid = calib_corners.at(tcid);

        bool success = calib_opt->initializeIntrinsics(
            j, cid.corners, cid.corner_ids, april_grid.aprilgrid_corner_pos_3d,
            img_vec[j].img->w, img_vec[j].img->h);
        if (success) break;
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

    Eigen::vector<Eigen::Vector2i> res;

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

  CalibHelper::initCamPoses(calib_opt->calib, this->vio_dataset,
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

  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];

    TimeCamId tcid0 = std::make_pair(timestamp_ns, 0);

    if (calib_init_poses.find(tcid0) == calib_init_poses.end()) continue;

    Sophus::SE3d T_a_c0 = calib_init_poses.at(tcid0).T_a_c;

    bool success = true;

    for (size_t j = 1; j < vio_dataset->get_num_cams(); j++) {
      TimeCamId tcid = std::make_pair(timestamp_ns, j);

      auto cd = calib_init_poses.find(tcid);
      if (cd != calib_init_poses.end() && cd->second.num_inliers > 0) {
        calib_opt->calib->T_i_c[j] = T_a_c0.inverse() * cd->second.T_a_c;
      } else {
        success = false;
      }
    }

    if (success) break;
  }

  std::cout << "Done camera extrinsics initialization:" << std::endl;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    std::cout << "T_c0_c" << j << ":\n"
              << calib_opt->calib->T_i_c[j].matrix() << std::endl;
  }
}

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

  std::set<uint64_t> invalid_timestamps;
  for (const auto &kv : calib_corners) {
    if (kv.second.corner_ids.size() < MIN_CORNERS)
      invalid_timestamps.insert(kv.first.first);
  }

  for (const auto &kv : calib_corners) {
    if (invalid_timestamps.find(kv.first.first) == invalid_timestamps.end())
      calib_opt->addAprilgridMeasurement(kv.first.first, kv.first.second,
                                         kv.second.corners,
                                         kv.second.corner_ids);
  }

  for (size_t j = 0; j < vio_dataset->get_image_timestamps().size(); ++j) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[j];

    for (size_t cam_id = 0; cam_id < calib_opt->calib->T_i_c.size(); cam_id++) {
      TimeCamId tcid = std::make_pair(timestamp_ns, cam_id);
      const auto cp_it = calib_init_poses.find(tcid);

      if (cp_it != calib_init_poses.end()) {
        calib_opt->addPoseMeasurement(
            timestamp_ns,
            cp_it->second.T_a_c * calib_opt->calib->T_i_c[cam_id].inverse());
        break;
      }
    }
  }

  calib_opt->init();
  computeProjections();

  std::cout << "Initialized optimization." << std::endl;
}

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

void CamCalib::optimizeWithParam(bool print_info,
                                 std::map<std::string, double> *stats) {
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

  if (calib_opt) {
    // calib_opt->compute_projections();
    double error;
    double reprojection_error;
    int num_points;

    auto start = std::chrono::high_resolution_clock::now();

    calib_opt->optimize(opt_intr, huber_thresh, error, num_points,
                        reprojection_error);

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

      std::cout << "==================================" << std::endl;
    }

    if (show_gui) {
      computeProjections();
    }
  }
}

void CamCalib::saveCalib() {
  if (calib_opt) {
    calib_opt->saveCalib(cache_path, vio_dataset->get_mocap_to_imu_offset_ns());

    std::cout << "Saved calibration in " << cache_path << "calibration.json"
              << std::endl;
  }
}

void CamCalib::drawImageOverlay(pangolin::View &v, size_t cam_id) {
  UNUSED(v);

  size_t frame_id = show_frame;

  if (vio_dataset && frame_id < vio_dataset->get_image_timestamps().size()) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[frame_id];
    TimeCamId tcid = std::make_pair(timestamp_ns, cam_id);

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

          for (size_t i = 0; i < rc.size(); i++) {
            Eigen::Vector2d c = rc[i];
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

          for (size_t i = 0; i < rc.size(); i++) {
            Eigen::Vector2d c = rc[i].head<2>();
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
