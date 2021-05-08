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
#include <basalt/spline/se3_spline.h>
#include <basalt/utils/sim_utils.h>
#include <basalt/vi_estimator/keypoint_vio.h>

#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/vis_utils.h>

// GUI functions
void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene();
void load_data(const std::string& calib_path);
void gen_data();
void compute_projections();
void setup_vio();
void draw_plots();
bool next_step();
void alignButton();

static const int knot_time = 3;
// static const double obs_std_dev = 0.5;

Eigen::Vector3d g(0, 0, -9.81);

// std::random_device rd{};
// std::mt19937 gen{rd()};
std::mt19937 gen{1};

// Simulated data

basalt::Se3Spline<5> gt_spline(int64_t(knot_time * 1e9));

Eigen::aligned_vector<Eigen::Vector3d> gt_points;
Eigen::aligned_vector<Sophus::SE3d> gt_frame_T_w_i;
Eigen::aligned_vector<Eigen::Vector3d> gt_frame_t_w_i, vio_t_w_i;
std::vector<int64_t> gt_frame_t_ns, kf_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> gt_accel, gt_gyro, gt_accel_bias,
    gt_gyro_bias, noisy_accel, noisy_gyro, gt_vel;
std::vector<int64_t> gt_imu_t_ns;

std::map<basalt::TimeCamId, basalt::SimObservations> gt_observations;
std::map<basalt::TimeCamId, basalt::SimObservations> noisy_observations;

std::string marg_data_path;

// VIO vars
basalt::Calibration<double> calib;
basalt::KeypointVioEstimator::Ptr vio;

// Visualization vars
std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;
tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
    out_state_queue;

std::vector<pangolin::TypedImage> images;

// Pangolin vars
constexpr int UI_WIDTH = 200;
pangolin::DataLog imu_data_log, vio_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1000);

pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_obs_noisy("ui.show_obs_noisy", true, false, true);
pangolin::Var<bool> show_obs_vio("ui.show_obs_vio", true, false, true);

pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<bool> show_accel("ui.show_accel", false, false, true);
pangolin::Var<bool> show_gyro("ui.show_gyro", false, false, true);
pangolin::Var<bool> show_gt_vel("ui.show_gt_vel", false, false, true);
pangolin::Var<bool> show_gt_pos("ui.show_gt_pos", true, false, true);
pangolin::Var<bool> show_gt_bg("ui.show_gt_bg", false, false, true);
pangolin::Var<bool> show_gt_ba("ui.show_gt_ba", false, false, true);

pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

pangolin::Var<bool> continue_btn("ui.continue", true, false, true);

Button align_step_btn("ui.align_se3", &alignButton);

int main(int argc, char** argv) {
  srand(1);

  bool show_gui = true;
  std::string cam_calib_path;
  std::string result_path;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--marg-data", marg_data_path,
                 "Folder to store marginalization data.")
      ->required();

  app.add_option("--result-path", result_path,
                 "Path to result file where the system will write RMSE ATE.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(cam_calib_path);

  gen_data();

  setup_vio();

  vio->out_vis_queue = &out_vis_queue;
  vio->out_state_queue = &out_state_queue;

  std::thread t0([&]() {
    for (size_t i = 0; i < gt_imu_t_ns.size(); i++) {
      basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
      data->t_ns = gt_imu_t_ns[i];

      data->accel = noisy_accel[i];
      data->gyro = noisy_gyro[i];

      vio->addIMUToQueue(data);
    }

    vio->addIMUToQueue(nullptr);

    std::cout << "Finished t0" << std::endl;
  });

  std::thread t1([&]() {
    for (const auto& t_ns : kf_t_ns) {
      basalt::OpticalFlowResult::Ptr data(new basalt::OpticalFlowResult);
      data->t_ns = t_ns;

      for (size_t j = 0; j < calib.T_i_c.size(); j++) {
        data->observations.emplace_back();
        basalt::TimeCamId tcid(data->t_ns, j);
        const basalt::SimObservations& obs = noisy_observations.at(tcid);
        for (size_t k = 0; k < obs.pos.size(); k++) {
          Eigen::AffineCompact2f t;
          t.setIdentity();
          t.translation() = obs.pos[k].cast<float>();
          data->observations.back()[obs.id[k]] = t;
        }
      }

      vio->addVisionToQueue(data);
    }

    vio->addVisionToQueue(nullptr);

    std::cout << "Finished t1" << std::endl;
  });

  std::thread t2([&]() {
    basalt::VioVisualizationData::Ptr data;

    while (true) {
      out_vis_queue.pop(data);

      if (data.get()) {
        vis_map[data->t_ns] = data;
      } else {
        break;
      }
    }

    std::cout << "Finished t2" << std::endl;
  });

  std::thread t3([&]() {
    basalt::PoseVelBiasState<double>::Ptr data;

    while (true) {
      out_state_queue.pop(data);

      if (!data.get()) break;

      int64_t t_ns = data->t_ns;

      // std::cerr << "t_ns " << t_ns << std::endl;
      Sophus::SE3d T_w_i = data->T_w_i;
      Eigen::Vector3d vel_w_i = data->vel_w_i;
      Eigen::Vector3d bg = data->bias_gyro;
      Eigen::Vector3d ba = data->bias_accel;

      vio_t_w_i.emplace_back(T_w_i.translation());

      {
        std::vector<float> vals;
        vals.push_back(t_ns * 1e-9);

        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

        vio_data_log.Log(vals);
      }
    }

    std::cout << "Finished t3" << std::endl;
  });

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.5)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    plotter = new pangolin::Plotter(&imu_data_log, 0.0, kf_t_ns.back() * 1e-9,
                                    -10.0, 10.0, 0.01f, 0.01f);
    plot_display.AddDisplay(*plotter);

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

    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(15, 3, 15, 0, 0, 0, pangolin::AxisZ));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.4, 1.0, 0.5, 1.0)
            .SetHandler(new pangolin::Handler3D(camera));

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);

      draw_scene();

      img_view_display.Activate();

      if (show_frame.GuiChanged()) {
        for (size_t i = 0; i < calib.intrinsics.size(); i++) {
          // img_view[i]->SetImage(images[i]);
        }
        draw_plots();
      }

      if (show_accel.GuiChanged() || show_gyro.GuiChanged() ||
          show_gt_vel.GuiChanged() || show_gt_pos.GuiChanged() ||
          show_gt_ba.GuiChanged() || show_gt_bg.GuiChanged() ||
          show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
          show_est_ba.GuiChanged() || show_est_bg.GuiChanged()) {
        draw_plots();
      }

      pangolin::FinishFrame();

      if (continue_btn) {
        if (!next_step()) continue_btn = false;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  }

  t0.join();
  t1.join();
  t2.join();
  t3.join();
  // t4.join();

  if (!result_path.empty()) {
    Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

    auto it = vis_map.find(kf_t_ns.back());

    if (it != vis_map.end()) {
      for (const auto& t : it->second->states)
        vio_t_w_i.emplace_back(t.translation());

    } else {
      std::cerr << "Could not find results!!" << std::endl;
    }

    BASALT_ASSERT(kf_t_ns.size() == vio_t_w_i.size());

    double error =
        basalt::alignSVD(kf_t_ns, vio_t_w_i, gt_frame_t_ns, gt_frame_t_w_i);

    std::ofstream os(result_path);
    os << error << std::endl;
    os.close();
  }

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);

  size_t frame_id = show_frame;
  basalt::TimeCamId tcid(kf_t_ns[frame_id], cam_id);

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (gt_observations.find(tcid) != gt_observations.end()) {
      const basalt::SimObservations& cr = gt_observations.at(tcid);

      for (size_t i = 0; i < cr.pos.size(); i++) {
        const float radius = 2;
        const Eigen::Vector2f c = cr.pos[i].cast<float>();
        pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

        if (show_ids)
          pangolin::GlFont::I().Text("%d", cr.id[i]).Draw(c[0], c[1]);
      }

      pangolin::GlFont::I().Text("%d gt points", cr.pos.size()).Draw(5, 20);
    }
  }

  if (show_obs_noisy) {
    glLineWidth(1.0);
    glColor3f(1.0, 1.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (noisy_observations.find(tcid) != noisy_observations.end()) {
      const basalt::SimObservations& cr = noisy_observations.at(tcid);

      for (size_t i = 0; i < cr.pos.size(); i++) {
        const float radius = 2;
        const Eigen::Vector2f c = cr.pos[i].cast<float>();
        pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

        if (show_ids)
          pangolin::GlFont::I().Text("%d", cr.id[i]).Draw(c[0], c[1]);
      }

      pangolin::GlFont::I().Text("%d noisy points", cr.pos.size()).Draw(5, 40);
    }
  }

  if (show_obs_vio) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto it = vis_map.find(gt_frame_t_ns[frame_id]);

    if (it != vis_map.end() && cam_id < it->second->projections.size()) {
      const auto& points = it->second->projections[cam_id];

      if (points.size() > 0) {
        double min_id = points[0][2], max_id = points[0][2];
        for (size_t i = 0; i < points.size(); i++) {
          min_id = std::min(min_id, points[i][2]);
          max_id = std::max(max_id, points[i][2]);
        }

        for (size_t i = 0; i < points.size(); i++) {
          const float radius = 2;
          const Eigen::Vector4d c = points[i];
          pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

          if (show_ids)
            pangolin::GlFont::I().Text("%d", int(c[3])).Draw(c[0], c[1]);
        }
      }

      glColor3f(0.0, 0.0, 1.0);
      pangolin::GlFont::I().Text("%d vio points", points.size()).Draw(5, 60);
    }
  }
}

void draw_scene() {
  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(gt_color);
  pangolin::glDrawPoints(gt_points);
  pangolin::glDrawLineStrip(gt_frame_t_w_i);

  glColor3ubv(cam_color);
  pangolin::glDrawLineStrip(vio_t_w_i);

  size_t frame_id = show_frame;

  auto it = vis_map.find(kf_t_ns[frame_id]);

  if (it != vis_map.end()) {
    for (const auto& p : it->second->states)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, cam_color, 0.1f);

    for (const auto& p : it->second->frames)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, pose_color, 0.1f);

    glColor3ubv(pose_color);
    pangolin::glDrawPoints(it->second->points);
  }

  // pangolin::glDrawAxis(gt_frame_T_w_i[frame_id].matrix(), 0.1);

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}

void gen_data() {
  // Save spline data
  {
    std::string path = marg_data_path + "/gt_spline.cereal";

    std::cout << "Loading gt_spline " << path << std::endl;

    std::ifstream is(path, std::ios::binary);
    {
      cereal::JSONInputArchive archive(is);

      int64_t t_ns;
      Eigen::aligned_vector<Sophus::SE3d> knots;

      archive(cereal::make_nvp("t_ns", t_ns));
      archive(cereal::make_nvp("knots", knots));

      gt_spline = basalt::Se3Spline<5>(t_ns);

      for (size_t i = 0; i < knots.size(); i++) {
        gt_spline.knotsPushBack(knots[i]);
      }

      archive(cereal::make_nvp("noisy_accel", noisy_accel));
      archive(cereal::make_nvp("noisy_gyro", noisy_gyro));
      archive(cereal::make_nvp("noisy_accel", gt_accel));
      archive(cereal::make_nvp("gt_gyro", gt_gyro));
      archive(cereal::make_nvp("gt_accel_bias", gt_accel_bias));
      archive(cereal::make_nvp("gt_gyro_bias", gt_gyro_bias));

      archive(cereal::make_nvp("gt_points", gt_points));

      archive(cereal::make_nvp("gt_observations", gt_observations));
      archive(cereal::make_nvp("noisy_observations", noisy_observations));

      archive(cereal::make_nvp("gt_points", gt_points));

      archive(cereal::make_nvp("gt_frame_t_ns", gt_frame_t_ns));
      archive(cereal::make_nvp("gt_imu_t_ns", gt_imu_t_ns));
    }

    gt_frame_t_w_i.clear();
    for (int64_t t_ns : gt_frame_t_ns) {
      gt_frame_t_w_i.emplace_back(gt_spline.pose(t_ns).translation());
    }

    is.close();
  }

  basalt::MargDataLoader mdl;
  tbb::concurrent_bounded_queue<basalt::MargData::Ptr> marg_queue;
  mdl.out_marg_queue = &marg_queue;

  mdl.start(marg_data_path);

  Eigen::aligned_map<int64_t, Sophus::SE3d> tmp_poses;

  while (true) {
    basalt::MargData::Ptr data;
    marg_queue.pop(data);

    if (data.get()) {
      for (const auto& kv : data->frame_poses) {
        tmp_poses[kv.first] = kv.second.getPose();
      }

      for (const auto& kv : data->frame_states) {
        if (data->kfs_all.count(kv.first) > 0) {
          tmp_poses[kv.first] = kv.second.getState().T_w_i;
        }
      }

    } else {
      break;
    }
  }

  for (const auto& kv : tmp_poses) {
    kf_t_ns.emplace_back(kv.first);
  }

  show_frame.Meta().range[1] = kf_t_ns.size() - 1;
}

void draw_plots() {
  plotter->ClearSeries();
  plotter->ClearMarkers();

  if (show_accel) {
    plotter->AddSeries("$0", "$1", pangolin::DrawingModeDashed,
                       pangolin::Colour::Red(), "accel measurements x");
    plotter->AddSeries("$0", "$2", pangolin::DrawingModeDashed,
                       pangolin::Colour::Green(), "accel measurements y");
    plotter->AddSeries("$0", "$3", pangolin::DrawingModeDashed,
                       pangolin::Colour::Blue(), "accel measurements z");
  }

  if (show_gyro) {
    plotter->AddSeries("$0", "$4", pangolin::DrawingModeDashed,
                       pangolin::Colour::Red(), "gyro measurements x");
    plotter->AddSeries("$0", "$5", pangolin::DrawingModeDashed,
                       pangolin::Colour::Green(), "gyro measurements y");
    plotter->AddSeries("$0", "$6", pangolin::DrawingModeDashed,
                       pangolin::Colour::Blue(), "gyro measurements z");
  }

  if (show_gt_vel) {
    plotter->AddSeries("$0", "$7", pangolin::DrawingModeDashed,
                       pangolin::Colour::Red(), "ground-truth velocity x");
    plotter->AddSeries("$0", "$8", pangolin::DrawingModeDashed,
                       pangolin::Colour::Green(), "ground-truth velocity y");
    plotter->AddSeries("$0", "$9", pangolin::DrawingModeDashed,
                       pangolin::Colour::Blue(), "ground-truth velocity z");
  }

  if (show_gt_pos) {
    plotter->AddSeries("$0", "$10", pangolin::DrawingModeDashed,
                       pangolin::Colour::Red(), "ground-truth position x");
    plotter->AddSeries("$0", "$11", pangolin::DrawingModeDashed,
                       pangolin::Colour::Green(), "ground-truth position y");
    plotter->AddSeries("$0", "$12", pangolin::DrawingModeDashed,
                       pangolin::Colour::Blue(), "ground-truth position z");
  }

  if (show_gt_bg) {
    plotter->AddSeries("$0", "$13", pangolin::DrawingModeDashed,
                       pangolin::Colour::Red(), "ground-truth gyro bias x");
    plotter->AddSeries("$0", "$14", pangolin::DrawingModeDashed,
                       pangolin::Colour::Green(), "ground-truth gyro bias y");
    plotter->AddSeries("$0", "$15", pangolin::DrawingModeDashed,
                       pangolin::Colour::Blue(), "ground-truth gyro bias z");
  }

  if (show_gt_ba) {
    plotter->AddSeries("$0", "$16", pangolin::DrawingModeDashed,
                       pangolin::Colour::Red(), "ground-truth accel bias x");
    plotter->AddSeries("$0", "$17", pangolin::DrawingModeDashed,
                       pangolin::Colour::Green(), "ground-truth accel bias y");
    plotter->AddSeries("$0", "$18", pangolin::DrawingModeDashed,
                       pangolin::Colour::Blue(), "ground-truth accel bias z");
  }

  if (show_est_vel) {
    plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "estimated velocity x",
                       &vio_data_log);
    plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "estimated velocity y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "estimated velocity z",
                       &vio_data_log);
  }

  if (show_est_pos) {
    plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "estimated position x",
                       &vio_data_log);
    plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "estimated position y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "estimated position z",
                       &vio_data_log);
  }

  if (show_est_bg) {
    plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "estimated gyro bias x",
                       &vio_data_log);
    plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "estimated gyro bias y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "estimated gyro bias z",
                       &vio_data_log);
  }

  if (show_est_ba) {
    plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "estimated accel bias x",
                       &vio_data_log);
    plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "estimated accel bias y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "estimated accel bias z",
                       &vio_data_log);
  }

  double t = kf_t_ns[show_frame] * 1e-9;
  plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                     pangolin::Colour::White());
}

void setup_vio() {
  int64_t t_init_ns = kf_t_ns.front();
  Sophus::SE3d T_w_i_init = gt_spline.pose(t_init_ns);
  Eigen::Vector3d vel_w_i_init = gt_spline.transVelWorld(t_init_ns);

  std::cout << "Setting up filter: t_ns " << t_init_ns << std::endl;
  std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;
  std::cout << "vel_w_i " << vel_w_i_init.transpose() << std::endl;

  basalt::VioConfig config;
  config.vio_debug = true;

  vio.reset(new basalt::KeypointVioEstimator(g, calib, config));
  vio->initialize(t_init_ns, T_w_i_init, vel_w_i_init, gt_gyro_bias.front(),
                  gt_accel_bias.front());

  vio->setMaxStates(10000);
  vio->setMaxKfs(10000);

  // int iteration = 0;
  vio_data_log.Clear();
  error_data_log.Clear();
  vio_t_w_i.clear();
}

bool next_step() {
  if (show_frame < int(kf_t_ns.size()) - 1) {
    show_frame = show_frame + 1;
    show_frame.Meta().gui_changed = true;
    return true;
  } else {
    return false;
  }
}

void alignButton() {
  Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

  auto it = vis_map.find(kf_t_ns.back());

  if (it != vis_map.end()) {
    for (const auto& t : it->second->states)
      vio_t_w_i.emplace_back(t.translation());

  } else {
    std::cerr << "Could not find results!!" << std::endl;
  }

  BASALT_ASSERT(kf_t_ns.size() == vio_t_w_i.size());

  basalt::alignSVD(kf_t_ns, vio_t_w_i, gt_frame_t_ns, gt_frame_t_w_i);
}
