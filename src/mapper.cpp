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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/optimization/accumulator.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/utils/filesystem.h>
#include <basalt/utils/imu_types.h>
#include <basalt/utils/nfr.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/vi_estimator/nfr_mapper.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

using basalt::POSE_SIZE;
using basalt::POSE_VEL_BIAS_SIZE;

Eigen::Vector3d g(0, 0, -9.81);

const Eigen::aligned_vector<Eigen::Vector2i> image_resolutions = {{752, 480},
                                                                  {752, 480}};

basalt::VioConfig vio_config;
basalt::NfrMapper::Ptr nrf_mapper;

Eigen::aligned_vector<Eigen::Vector3d> gt_frame_t_w_i;
std::vector<int64_t> gt_frame_t_ns, image_t_ns;

Eigen::aligned_vector<Eigen::Vector3d> mapper_points;
std::vector<int> mapper_point_ids;

std::map<int64_t, basalt::MargData::Ptr> marg_data;

Eigen::aligned_vector<Eigen::Vector3d> edges_vis;
Eigen::aligned_vector<Eigen::Vector3d> roll_pitch_vis;
Eigen::aligned_vector<Eigen::Vector3d> rel_edges_vis;

void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene();
void load_data(const std::string& calib_path,
               const std::string& marg_data_path);
void processMargData(basalt::MargData& m);
void extractNonlinearFactors(basalt::MargData& m);
void computeEdgeVis();
void optimize();
void randomInc();
void randomYawInc();
void compute_error();
double alignButton();
void detect();
void match();
void tracks();
void optimize();
void filter();
void saveTrajectoryButton();

constexpr int UI_WIDTH = 200;

basalt::Calibration<double> calib;

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, 0);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1);
pangolin::Var<int> show_cam2("ui.show_cam2", 0, 0, 0);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, false, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, false, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, false, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<int> num_opt_iter("ui.num_opt_iter", 10, 0, 20);

pangolin::Var<bool> show_gt("ui.show_gt", true, false, true);
pangolin::Var<bool> show_edges("ui.show_edges", true, false, true);
pangolin::Var<bool> show_points("ui.show_points", true, false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button detect_btn("ui.detect", &detect);
Button match_btn("ui.match", &match);
Button tracks_btn("ui.tracks", &tracks);
Button optimize_btn("ui.optimize", &optimize);

pangolin::Var<double> outlier_threshold("ui.outlier_threshold", 3.0, 0.01, 10);

Button filter_btn("ui.filter", &filter);
Button align_btn("ui.aling_se3", &alignButton);

pangolin::Var<bool> euroc_fmt("ui.euroc_fmt", true, false, true);
pangolin::Var<bool> tum_rgbd_fmt("ui.tum_rgbd_fmt", false, false, true);
Button save_traj_btn("ui.save_traj", &saveTrajectoryButton);

pangolin::OpenGlRenderState camera;

std::string marg_data_path;

int main(int argc, char** argv) {
  bool show_gui = true;
  std::string cam_calib_path;
  std::string result_path;
  std::string config_path;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--marg-data", marg_data_path, "Path to cache folder.")
      ->required();

  app.add_option("--config-path", config_path, "Path to config file.");

  app.add_option("--result-path", result_path, "Path to config file.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (!config_path.empty()) {
    vio_config.load(config_path);
  }

  load_data(cam_calib_path, marg_data_path);

  for (auto& kv : marg_data) {
    nrf_mapper->addMargData(kv.second);
  }

  computeEdgeVis();

  {
    std::cout << "Loaded " << nrf_mapper->img_data.size() << " images."
              << std::endl;

    show_frame1.Meta().range[1] = nrf_mapper->img_data.size() - 1;
    show_frame2.Meta().range[1] = nrf_mapper->img_data.size() - 1;
    show_frame1.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;

    show_cam1.Meta().range[1] = calib.intrinsics.size() - 1;
    show_cam2.Meta().range[1] = calib.intrinsics.size() - 1;
    if (calib.intrinsics.size() > 1) show_cam2 = 1;

    for (const auto& kv : nrf_mapper->img_data) {
      image_t_ns.emplace_back(kv.first);
    }

    std::sort(image_t_ns.begin(), image_t_ns.end());
  }

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.4)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    camera = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    //    pangolin::OpenGlRenderState camera(
    //        pangolin::ProjectionMatrixOrthographic(-30, 30, -30, 30, -30, 30),
    //        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
    //                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.0, 1.0, 0.4, 1.0)
            .SetHandler(new pangolin::Handler3D(camera));

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          show_frame2 = show_frame1;
          show_frame2.Meta().gui_changed = true;
          show_frame1.Meta().gui_changed = true;
        } else if (show_frame2.GuiChanged()) {
          show_frame1 = show_frame2;
          show_frame1.Meta().gui_changed = true;
          show_frame2.Meta().gui_changed = true;
        }
      }

      display3D.Activate(camera);
      glClearColor(1.f, 1.f, 1.f, 1.0f);

      draw_scene();

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(show_frame1);
        int64_t timestamp = image_t_ns[frame_id];
        size_t cam_id = show_cam1;

        if (nrf_mapper->img_data.count(timestamp) > 0 &&
            nrf_mapper->img_data.at(timestamp).get()) {
          const std::vector<basalt::ImageData>& img_vec =
              nrf_mapper->img_data.at(timestamp)->img_data;

          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_SHORT;
          fmt.scalable_internal_format = GL_LUMINANCE16;

          if (img_vec[cam_id].img.get()) {
            img_view[0]->SetImage(
                img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
          } else {
            img_view[0]->Clear();
          }
        } else {
          img_view[0]->Clear();
        }
      }

      if (euroc_fmt.GuiChanged()) {
        tum_rgbd_fmt = !euroc_fmt;
      }

      if (tum_rgbd_fmt.GuiChanged()) {
        euroc_fmt = !tum_rgbd_fmt;
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(show_frame2);
        int64_t timestamp = image_t_ns[frame_id];
        size_t cam_id = show_cam2;

        if (nrf_mapper->img_data.count(timestamp) > 0 &&
            nrf_mapper->img_data.at(timestamp).get()) {
          const std::vector<basalt::ImageData>& img_vec =
              nrf_mapper->img_data.at(timestamp)->img_data;

          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_SHORT;
          fmt.scalable_internal_format = GL_LUMINANCE16;

          if (img_vec[cam_id].img.get()) {
            img_view[1]->SetImage(
                img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
          } else {
            img_view[1]->Clear();
          }
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  } else {
    auto time_start = std::chrono::high_resolution_clock::now();
    // optimize();
    detect();
    match();
    tracks();
    optimize();
    filter();
    optimize();

    auto time_end = std::chrono::high_resolution_clock::now();

    if (!result_path.empty()) {
      double error = alignButton();

      auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          time_end - time_start);

      std::ofstream os(result_path);
      {
        cereal::JSONOutputArchive ar(os);
        ar(cereal::make_nvp("rms_ate", error));
        ar(cereal::make_nvp("num_frames", nrf_mapper->getFramePoses().size()));
        ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
      }
      os.close();
    }
  }

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  size_t frame_id = (view_id == 0) ? show_frame1 : show_frame2;
  size_t cam_id = (view_id == 0) ? show_cam1 : show_cam2;

  basalt::TimeCamId tcid(image_t_ns[frame_id], cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (nrf_mapper->feature_corners.find(tcid) !=
        nrf_mapper->feature_corners.end()) {
      const basalt::KeypointsData& cr = nrf_mapper->feature_corners.at(tcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, 20);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    size_t o_frame_id = (view_id == 0 ? show_frame2 : show_frame1);
    size_t o_cam_id = (view_id == 0 ? show_cam2 : show_cam1);

    basalt::TimeCamId o_tcid(image_t_ns[o_frame_id], o_cam_id);

    int idx = -1;

    auto it = nrf_mapper->feature_matches.find(std::make_pair(tcid, o_tcid));

    if (it != nrf_mapper->feature_matches.end()) {
      idx = 0;
    } else {
      it = nrf_mapper->feature_matches.find(std::make_pair(o_tcid, tcid));
      if (it != nrf_mapper->feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (nrf_mapper->feature_corners.find(tcid) !=
          nrf_mapper->feature_corners.end()) {
        const basalt::KeypointsData& cr = nrf_mapper->feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (nrf_mapper->feature_corners.find(tcid) !=
          nrf_mapper->feature_corners.end()) {
        const basalt::KeypointsData& cr = nrf_mapper->feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }
}

void draw_scene() {
  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(pose_color);
  if (show_points) pangolin::glDrawPoints(mapper_points);

  glColor3ubv(gt_color);
  if (show_gt) pangolin::glDrawLineStrip(gt_frame_t_w_i);

  glColor3f(0.0, 1.0, 0.0);
  if (show_edges) pangolin::glDrawLines(edges_vis);

  glLineWidth(2);
  glColor3f(1.0, 0.0, 1.0);
  if (show_edges) pangolin::glDrawLines(roll_pitch_vis);
  glLineWidth(1);

  glColor3f(1.0, 0.0, 0.0);
  if (show_edges) pangolin::glDrawLines(rel_edges_vis);

  for (const auto& kv : nrf_mapper->getFramePoses()) {
    pangolin::glDrawAxis(kv.second.getPose().matrix(), 0.1);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path, const std::string& cache_path) {
  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
      std::cout << "Loaded camera with " << calib.intrinsics.size()
                << " cameras" << std::endl;

    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  {
    // Load gt.
    {
      std::string p = cache_path + "/gt.cereal";
      std::ifstream is(p, std::ios::binary);

      {
        cereal::BinaryInputArchive archive(is);
        archive(gt_frame_t_ns);
        archive(gt_frame_t_w_i);
      }
      is.close();
      std::cout << "Loaded " << gt_frame_t_ns.size() << " timestamps and "
                << gt_frame_t_w_i.size() << " poses" << std::endl;
    }
  }

  nrf_mapper.reset(new basalt::NfrMapper(calib, vio_config));

  basalt::MargDataLoader mdl;
  tbb::concurrent_bounded_queue<basalt::MargData::Ptr> marg_queue;
  mdl.out_marg_queue = &marg_queue;

  mdl.start(cache_path);

  while (true) {
    basalt::MargData::Ptr data;
    marg_queue.pop(data);

    if (data.get()) {
      int64_t t_ns = *data->kfs_to_marg.begin();
      marg_data[t_ns] = data;

    } else {
      break;
    }
  }

  std::cout << "Loaded " << marg_data.size() << " marg data." << std::endl;
}

void computeEdgeVis() {
  edges_vis.clear();
  for (const auto& kv1 : nrf_mapper->lmdb.getObservations()) {
    for (const auto& kv2 : kv1.second) {
      Eigen::Vector3d p1 = nrf_mapper->getFramePoses()
                               .at(kv1.first.frame_id)
                               .getPose()
                               .translation();
      Eigen::Vector3d p2 = nrf_mapper->getFramePoses()
                               .at(kv2.first.frame_id)
                               .getPose()
                               .translation();

      edges_vis.emplace_back(p1);
      edges_vis.emplace_back(p2);
    }
  }

  roll_pitch_vis.clear();
  for (const auto& v : nrf_mapper->roll_pitch_factors) {
    const Sophus::SE3d& T_w_i =
        nrf_mapper->getFramePoses().at(v.t_ns).getPose();

    Eigen::Vector3d p = T_w_i.translation();
    Eigen::Vector3d d =
        v.R_w_i_meas * T_w_i.so3().inverse() * (-Eigen::Vector3d::UnitZ());

    roll_pitch_vis.emplace_back(p);
    roll_pitch_vis.emplace_back(p + 0.1 * d);
  }

  rel_edges_vis.clear();
  for (const auto& v : nrf_mapper->rel_pose_factors) {
    Eigen::Vector3d p1 =
        nrf_mapper->getFramePoses().at(v.t_i_ns).getPose().translation();
    Eigen::Vector3d p2 =
        nrf_mapper->getFramePoses().at(v.t_j_ns).getPose().translation();

    rel_edges_vis.emplace_back(p1);
    rel_edges_vis.emplace_back(p2);
  }
}

void optimize() {
  nrf_mapper->optimize(num_opt_iter);
  nrf_mapper->get_current_points(mapper_points, mapper_point_ids);

  computeEdgeVis();
}

double alignButton() {
  Eigen::aligned_vector<Eigen::Vector3d> filter_t_w_i;
  std::vector<int64_t> filter_t_ns;

  for (const auto& kv : nrf_mapper->getFramePoses()) {
    filter_t_ns.emplace_back(kv.first);
    filter_t_w_i.emplace_back(kv.second.getPose().translation());
  }

  return basalt::alignSVD(filter_t_ns, filter_t_w_i, gt_frame_t_ns,
                          gt_frame_t_w_i);
}

void detect() {
  nrf_mapper->feature_corners.clear();
  nrf_mapper->feature_matches.clear();
  nrf_mapper->detect_keypoints();
}

void match() {
  nrf_mapper->feature_matches.clear();
  nrf_mapper->match_stereo();
  nrf_mapper->match_all();
}

void tracks() {
  nrf_mapper->build_tracks();
  nrf_mapper->setup_opt();
  nrf_mapper->get_current_points(mapper_points, mapper_point_ids);
  //  nrf_mapper->get_current_points_with_color(mapper_points,
  //  mapper_points_color,
  //                                            mapper_point_ids);
  computeEdgeVis();
}

void filter() {
  nrf_mapper->filterOutliers(outlier_threshold, 4);
  nrf_mapper->get_current_points(mapper_points, mapper_point_ids);
}

void saveTrajectoryButton() {
  if (tum_rgbd_fmt) {
    std::ofstream os("keyframeTrajectory.txt");

    os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

    for (const auto& kv : nrf_mapper->getFramePoses()) {
      const Sophus::SE3d pose = kv.second.getPose();
      os << std::scientific << std::setprecision(18) << kv.first * 1e-9 << " "
         << pose.translation().x() << " " << pose.translation().y() << " "
         << pose.translation().z() << " " << pose.unit_quaternion().x() << " "
         << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z()
         << " " << pose.unit_quaternion().w() << std::endl;
    }

    os.close();

    std::cout << "Saved trajectory in TUM RGB-D Dataset format in "
                 "keyframeTrajectory.txt"
              << std::endl;
  } else {
    std::ofstream os("keyframeTrajectory.csv");

    os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
          "[],q_RS_x [],q_RS_y [],q_RS_z []"
       << std::endl;

    for (const auto& kv : nrf_mapper->getFramePoses()) {
      const Sophus::SE3d pose = kv.second.getPose();
      os << std::scientific << std::setprecision(18) << kv.first << ","
         << pose.translation().x() << "," << pose.translation().y() << ","
         << pose.translation().z() << "," << pose.unit_quaternion().w() << ","
         << pose.unit_quaternion().x() << "," << pose.unit_quaternion().y()
         << "," << pose.unit_quaternion().z() << std::endl;
    }

    os.close();

    std::cout
        << "Saved trajectory in Euroc Dataset format in keyframeTrajectory.csv"
        << std::endl;
  }
}
