/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko, Michael Loipführer and Nikolaus Demmel.
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

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <thread>

#include <librealsense2/rs.hpp>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/concurrent_queue.h>

#include <basalt/device/rs_t265.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/filesystem.h>
#include <CLI/CLI.hpp>
#include <cereal/archives/json.hpp>

constexpr int UI_WIDTH = 200;

basalt::RsT265Device::Ptr t265_device;

std::shared_ptr<pangolin::DataLog> imu_log;

pangolin::Var<int> webp_quality("ui.webp_quality", 90, 0, 101);
pangolin::Var<int> skip_frames("ui.skip_frames", 1, 1, 10);
pangolin::Var<float> exposure("ui.exposure", 5.0, 1, 20);

tbb::concurrent_bounded_queue<basalt::OpticalFlowInput::Ptr> image_data_queue,
    image_data_queue2;
tbb::concurrent_bounded_queue<basalt::ImuData<double>::Ptr> imu_data_queue;
tbb::concurrent_bounded_queue<basalt::RsPoseData> pose_data_queue;

std::atomic<bool> stop_workers;
std::atomic<bool> recording;

std::string dataset_dir;

static constexpr int NUM_CAMS = basalt::RsT265Device::NUM_CAMS;
static constexpr int NUM_WORKERS = 8;

std::ofstream cam_data[NUM_CAMS], exposure_data[NUM_CAMS], imu0_data, pose_data;

std::vector<std::thread> worker_threads;
std::thread imu_worker_thread, pose_worker_thread, exposure_save_thread,
    stop_recording_thread;

#if CV_MAJOR_VERSION >= 3
std::string file_extension = ".webp";
#else
std::string file_extension = ".jpg";
#endif

// manual exposure mode, if not enabled will also record pose data
bool manual_exposure;

void exposure_save_worker() {
  basalt::OpticalFlowInput::Ptr img;
  while (!stop_workers) {
    if (image_data_queue.try_pop(img)) {
      for (size_t cam_id = 0; cam_id < NUM_CAMS; ++cam_id) {
        cam_data[cam_id] << img->t_ns << "," << img->t_ns << file_extension
                         << std::endl;

        exposure_data[cam_id] << img->t_ns << ","
                              << int64_t(img->img_data[cam_id].exposure * 1e9)
                              << std::endl;
      }

      image_data_queue2.push(img);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void image_save_worker() {
  basalt::OpticalFlowInput::Ptr img;

  while (!stop_workers) {
    if (image_data_queue2.try_pop(img)) {
      for (size_t cam_id = 0; cam_id < NUM_CAMS; ++cam_id) {
        basalt::ManagedImage<uint16_t>::Ptr image_raw =
            img->img_data[cam_id].img;

        if (!image_raw.get()) continue;

        cv::Mat image(image_raw->h, image_raw->w, CV_8U);

        uint8_t *dst = image.ptr();
        const uint16_t *src = image_raw->ptr;

        for (size_t i = 0; i < image_raw->size(); i++) {
          dst[i] = (src[i] >> 8);
        }

#if CV_MAJOR_VERSION >= 3
        std::string filename = dataset_dir + "mav0/cam" +
                               std::to_string(cam_id) + "/data/" +
                               std::to_string(img->t_ns) + ".webp";

        std::vector<int> compression_params = {cv::IMWRITE_WEBP_QUALITY,
                                               webp_quality};
        cv::imwrite(filename, image, compression_params);
#else
        std::string filename = dataset_dir + "mav0/cam" +
                               std::to_string(cam_id) + "/data/" +
                               std::to_string(img->t_ns) + ".jpg";

        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY,
                                               webp_quality};
        cv::imwrite(filename, image, compression_params);
#endif
      }

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void imu_save_worker() {
  basalt::ImuData<double>::Ptr data;

  while (!stop_workers) {
    if (imu_data_queue.try_pop(data)) {
      if (imu_log.get())
        imu_log->Log(data->accel[0], data->accel[1], data->accel[2]);

      if (recording) {
        imu0_data << data->t_ns << "," << data->gyro[0] << "," << data->gyro[1]
                  << "," << data->gyro[2] << "," << data->accel[0] << ","
                  << data->accel[1] << "," << data->accel[2] << "\n";
      }

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void pose_save_worker() {
  basalt::RsPoseData data;

  while (!stop_workers) {
    if (pose_data_queue.try_pop(data)) {
      if (recording) {
        pose_data << data.t_ns << "," << data.data.translation().x() << ","
                  << data.data.translation().y() << ","
                  << data.data.translation().z() << ","
                  << data.data.unit_quaternion().w() << ","
                  << data.data.unit_quaternion().x() << ","
                  << data.data.unit_quaternion().y() << ","
                  << data.data.unit_quaternion().z() << std::endl;
      }

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void save_calibration(const basalt::RsT265Device::Ptr &device) {
  auto calib = device->exportCalibration();

  if (calib) {
    std::ofstream os(dataset_dir + "/calibration.json");
    cereal::JSONOutputArchive archive(os);

    archive(*calib);
  }
}

inline std::string get_date() {
  constexpr int MAX_DATE = 64;
  time_t now;
  char the_date[MAX_DATE];

  the_date[0] = '\0';

  now = time(nullptr);

  if (now != -1) {
    strftime(the_date, MAX_DATE, "%Y_%m_%d_%H_%M_%S", gmtime(&now));
  }

  return std::string(the_date);
}

void startRecording(const std::string &dir_path) {
  if (!recording) {
    if (stop_recording_thread.joinable()) stop_recording_thread.join();

    dataset_dir = dir_path + "dataset_" + get_date() + "/";

    basalt::fs::create_directory(dataset_dir);
    basalt::fs::create_directory(dataset_dir + "mav0/");
    basalt::fs::create_directory(dataset_dir + "mav0/cam0/");
    basalt::fs::create_directory(dataset_dir + "mav0/cam0/data/");
    basalt::fs::create_directory(dataset_dir + "mav0/cam1/");
    basalt::fs::create_directory(dataset_dir + "mav0/cam1/data/");
    basalt::fs::create_directory(dataset_dir + "mav0/imu0/");

    cam_data[0].open(dataset_dir + "mav0/cam0/data.csv");
    cam_data[1].open(dataset_dir + "mav0/cam1/data.csv");
    exposure_data[0].open(dataset_dir + "mav0/cam0/exposure.csv");
    exposure_data[1].open(dataset_dir + "mav0/cam1/exposure.csv");
    imu0_data.open(dataset_dir + "mav0/imu0/data.csv");

    if (!manual_exposure) {
      basalt::fs::create_directory(dataset_dir + "mav0/realsense0/");
      pose_data.open(dataset_dir + "mav0/realsense0/data.csv");
      pose_data << "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
                   "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n";
    }

    cam_data[0] << "#timestamp [ns], filename\n";
    cam_data[1] << "#timestamp [ns], filename\n";
    exposure_data[0] << "#timestamp [ns], exposure time[ns]\n";
    exposure_data[1] << "#timestamp [ns], exposure time[ns]\n";
    imu0_data << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad "
                 "s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y "
                 "[m s^-2],a_RS_S_z [m s^-2]\n";

    save_calibration(t265_device);
    t265_device->image_data_queue = &image_data_queue;
    t265_device->imu_data_queue = &imu_data_queue;
    t265_device->pose_data_queue = &pose_data_queue;

    std::cout << "Started recording dataset in " << dataset_dir << std::endl;

    recording = true;
  } else {
    std::cout << "Already recording" << std::endl;
  }
}

void stopRecording() {
  if (recording) {
    auto stop_recording_func = [&]() {
      t265_device->imu_data_queue = nullptr;
      t265_device->pose_data_queue = nullptr;
      t265_device->image_data_queue = nullptr;

      while (!image_data_queue.empty() || !image_data_queue2.empty() ||
             !imu_data_queue.empty() || !pose_data_queue.empty()) {
        std::cout << "Waiting until the data from the queues is written to the "
                     "hard drive."
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }

      recording = false;
      cam_data[0].close();
      cam_data[1].close();
      exposure_data[0].close();
      exposure_data[1].close();
      imu0_data.close();
      pose_data.close();

      std::cout << "Stopped recording dataset in " << dataset_dir << std::endl;
    };

    stop_recording_thread = std::thread(stop_recording_func);
  }
}

void toggleRecording(const std::string &dir_path) {
  if (recording) {
    stopRecording();
  } else {
    startRecording(dir_path);
  }
}

int main(int argc, char *argv[]) {
  CLI::App app{"Record RealSense T265 Data"};

  std::string dataset_path;

  app.add_option("--dataset-path", dataset_path, "Path to dataset");
  app.add_flag("--manual-exposure", manual_exposure,
               "If set will enable manual exposure.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (!dataset_path.empty() && dataset_path[dataset_path.length() - 1] != '/') {
    dataset_path += '/';
  }

  bool show_gui = true;

  stop_workers = false;
  if (worker_threads.empty()) {
    for (int i = 0; i < NUM_WORKERS; i++) {
      worker_threads.emplace_back(image_save_worker);
    }
  }

  exposure_save_thread = std::thread(exposure_save_worker);
  imu_worker_thread = std::thread(imu_save_worker);
  pose_worker_thread = std::thread(pose_save_worker);

  image_data_queue.set_capacity(1000);
  image_data_queue2.set_capacity(1000);
  imu_data_queue.set_capacity(10000);
  pose_data_queue.set_capacity(10000);

  // realsense
  t265_device.reset(new basalt::RsT265Device(manual_exposure, skip_frames,
                                             webp_quality, exposure));

  t265_device->start();
  imu_log.reset(new pangolin::DataLog);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Record RealSense T265", 1200, 800);

    pangolin::Var<std::function<void(void)>> record_btn(
        "ui.record", [&] { return toggleRecording(dataset_path); });
    pangolin::Var<std::function<void(void)>> export_calibration(
        "ui.export_calib", [&] { return save_calibration(t265_device); });

    std::atomic<int64_t> record_t_ns;
    record_t_ns = 0;

    glEnable(GL_DEPTH_TEST);

    pangolin::View &img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::View &plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < basalt::RsT265Device::NUM_CAMS) {
      int idx = img_view.size();
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      iv->extern_draw_function = [&, idx](pangolin::View &v) {
        UNUSED(v);

        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);  // red
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (t265_device->last_img_data.get())
          pangolin::GlFont::I()
              .Text("Exposure: %.3f ms.",
                    t265_device->last_img_data->img_data[idx].exposure * 1000.0)
              .Draw(30, 30);

        if (idx == 0) {
          pangolin::GlFont::I()
              .Text("Queue: %d.", image_data_queue2.size())
              .Draw(30, 60);
        }

        if (idx == 0 && recording) {
          pangolin::GlFont::I().Text("Recording").Draw(30, 90);
        }
      };

      iv->OnSelectionCallback =
          [&](pangolin::ImageView::OnSelectionEventData o) {
            UNUSED(o);

            int64_t curr_t_ns = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();
            if (std::abs(record_t_ns - curr_t_ns) > int64_t(2e9)) {
              toggleRecording(dataset_path);
              record_t_ns = curr_t_ns;
            }
          };

      img_view.push_back(iv);
      img_view_display.AddDisplay(*iv);
    }

    imu_log->Clear();

    std::vector<std::string> labels;
    labels.push_back(std::string("accel x"));
    labels.push_back(std::string("accel y"));
    labels.push_back(std::string("accel z"));
    imu_log->SetLabels(labels);

    pangolin::Plotter plotter(imu_log.get(), 0.0f, 2000.0f, -15.0f, 15.0f, 0.1f,
                              0.1f);
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
    plotter.Track("$i");

    plot_display.AddDisplay(plotter);

    plotter.ClearSeries();
    plotter.AddSeries("$i", "$0", pangolin::DrawingModeLine,
                      pangolin::Colour::Red(), "accel x");
    plotter.AddSeries("$i", "$1", pangolin::DrawingModeLine,
                      pangolin::Colour::Green(), "accel y");
    plotter.AddSeries("$i", "$2", pangolin::DrawingModeLine,
                      pangolin::Colour::Blue(), "accel z");

    while (!pangolin::ShouldQuit()) {
      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_SHORT;
        fmt.scalable_internal_format = GL_LUMINANCE16;

        if (t265_device->last_img_data.get())
          for (size_t cam_id = 0; cam_id < basalt::RsT265Device::NUM_CAMS;
               cam_id++) {
            if (t265_device->last_img_data->img_data[cam_id].img.get())
              img_view[cam_id]->SetImage(
                  t265_device->last_img_data->img_data[cam_id].img->ptr,
                  t265_device->last_img_data->img_data[cam_id].img->w,
                  t265_device->last_img_data->img_data[cam_id].img->h,
                  t265_device->last_img_data->img_data[cam_id].img->pitch, fmt);
          }
      }

      if (manual_exposure && exposure.GuiChanged()) {
        t265_device->setExposure(exposure);
      }

      if (webp_quality.GuiChanged()) {
        t265_device->setWebpQuality(webp_quality);
      }

      if (skip_frames.GuiChanged()) {
        t265_device->setSkipFrames(skip_frames);
      }

      pangolin::FinishFrame();

      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
  }

  if (recording) stopRecording();
  stop_workers = true;

  for (auto &t : worker_threads) t.join();
  imu_worker_thread.join();
  pose_worker_thread.join();

  return EXIT_SUCCESS;
}
