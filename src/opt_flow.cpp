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
#include <basalt/spline/se3_spline.h>

#include <basalt/calibration/calibration.hpp>

#include <basalt/optical_flow/optical_flow.h>

#include <basalt/serialization/headers_serialization.h>

constexpr int UI_WIDTH = 200;

void draw_image_overlay(pangolin::View& v, size_t cam_id);
void load_data(const std::string& calib_path);
bool next_step();
bool prev_step();

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1500);
pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

using Button = pangolin::Var<std::function<void(void)>>;
Button next_step_btn("ui.next_step", &next_step);
Button prev_step_btn("ui.prev_step", &prev_step);
pangolin::Var<bool> continue_btn("ui.continue", true, false, true);

// Opt flow variables
basalt::VioDatasetPtr vio_dataset;

basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;

tbb::concurrent_unordered_map<int64_t, basalt::OpticalFlowResult::Ptr,
                              std::hash<int64_t>>
    observations;
tbb::concurrent_bounded_queue<basalt::OpticalFlowResult::Ptr>
    observations_queue;

basalt::Calibration<double> calib;

std::unordered_map<basalt::KeypointId, int> keypoint_stats;

void feed_images() {
  std::cout << "Started input_data thread " << std::endl;

  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);

    data->t_ns = vio_dataset->get_image_timestamps()[i];
    data->img_data = vio_dataset->get_image_data(data->t_ns);

    opt_flow_ptr->input_queue.push(data);
  }

  // Indicate the end of the sequence
  basalt::OpticalFlowInput::Ptr data;
  opt_flow_ptr->input_queue.push(data);

  std::cout << "Finished input_data thread " << std::endl;
}

void read_result() {
  std::cout << "Started read_result thread " << std::endl;

  basalt::OpticalFlowResult::Ptr res;

  while (true) {
    observations_queue.pop(res);
    if (!res.get()) break;

    res->input_images.reset();

    observations.emplace(res->t_ns, res);

    for (size_t i = 0; i < res->observations.size(); i++)
      for (const auto& kv : res->observations.at(i)) {
        if (keypoint_stats.count(kv.first) == 0) {
          keypoint_stats[kv.first] = 1;
        } else {
          keypoint_stats[kv.first]++;
        }
      }
  }

  std::cout << "Finished read_result thread " << std::endl;

  double sum = 0;

  for (const auto& kv : keypoint_stats) {
    sum += kv.second;
  }

  std::cout << "Mean track length: " << sum / keypoint_stats.size()
            << " num_points: " << keypoint_stats.size() << std::endl;
}

int main(int argc, char** argv) {
  bool show_gui = true;
  std::string cam_calib_path;
  std::string dataset_path;
  std::string dataset_type;
  std::string config_path;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--dataset-path", dataset_path, "Path to dataset.")
      ->required();

  app.add_option("--dataset-type", dataset_type, "Type of dataset.")
      ->required();

  app.add_option("--config-path", config_path, "Path to config file.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (!config_path.empty()) {
    vio_config.load(config_path);
  }

  load_data(cam_calib_path);

  {
    basalt::DatasetIoInterfacePtr dataset_io =
        basalt::DatasetIoFactory::getDatasetIo(dataset_type);

    dataset_io->read(dataset_path);

    vio_dataset = dataset_io->get_data();
    vio_dataset->get_image_timestamps().erase(
        vio_dataset->get_image_timestamps().begin());

    show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
    show_frame.Meta().gui_changed = true;

    opt_flow_ptr =
        basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);
    if (show_gui) opt_flow_ptr->output_queue = &observations_queue;
    observations_queue.set_capacity(100);

    keypoint_stats.reserve(50000);
  }

  std::thread t1(&feed_images);

  if (show_gui) {
    std::thread t2(&read_result);

    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
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

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);

      img_view_display.Activate();

      if (show_frame.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(show_frame);
        int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

        const std::vector<basalt::ImageData>& img_vec =
            vio_dataset->get_image_data(timestamp);

        for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
          if (img_vec[cam_id].img.get()) {
            auto img = img_vec[cam_id].img;

            pangolin::GlPixFormat fmt;
            fmt.glformat = GL_LUMINANCE;
            fmt.gltype = GL_UNSIGNED_SHORT;
            fmt.scalable_internal_format = GL_LUMINANCE16;

            img_view[cam_id]->SetImage(img->ptr, img->w, img->h, img->pitch,
                                       fmt);
          } else {
            img_view[cam_id]->Clear();
          }
        }
      }

      pangolin::FinishFrame();

      if (continue_btn) {
        if (!next_step()) {
          continue_btn = false;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

    t2.join();
  }

  t1.join();

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);

  size_t frame_id = static_cast<size_t>(show_frame);
  int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (observations.count(t_ns) > 0) {
      const Eigen::aligned_map<basalt::KeypointId, Eigen::AffineCompact2f>&
          kp_map = observations.at(t_ns)->observations[cam_id];

      for (const auto& kv : kp_map) {
        Eigen::MatrixXf transformed_patch =
            kv.second.linear() * opt_flow_ptr->patch_coord;
        transformed_patch.colwise() += kv.second.translation();

        for (int i = 0; i < transformed_patch.cols(); i++) {
          const Eigen::Vector2f c = transformed_patch.col(i);
          pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
        }

        const Eigen::Vector2f c = kv.second.translation();

        if (show_ids)
          pangolin::GlFont::I().Text("%d", kv.first).Draw(5 + c[0], 5 + c[1]);
      }

      pangolin::GlFont::I()
          .Text("Tracked %d keypoints", kp_map.size())
          .Draw(5, 20);
    }
  }
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

bool next_step() {
  if (show_frame < int(vio_dataset->get_image_timestamps().size()) - 1) {
    show_frame = show_frame + 1;
    show_frame.Meta().gui_changed = true;
    return true;
  } else {
    return false;
  }
}

bool prev_step() {
  if (show_frame > 0) {
    show_frame = show_frame - 1;
    show_frame.Meta().gui_changed = true;
    return true;
  } else {
    return false;
  }
}
