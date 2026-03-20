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

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <thread>

#include <librealsense2/rs.hpp>

#include <pangolin/display/default_font.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

static constexpr int NUM_CAMS = basalt::RsT265Device::NUM_CAMS;
static constexpr int NUM_WORKERS = 8;

std::vector<std::thread> worker_threads;
std::thread imu_worker_thread, pose_worker_thread, exposure_save_thread,
    stop_recording_thread;

struct RecordingSession {
  using Ptr = std::shared_ptr<RecordingSession>;

  std::string dataset_dir;
  std::ofstream cam_data[NUM_CAMS];
  std::ofstream exposure_data[NUM_CAMS];
  std::ofstream imu0_data;
  std::ofstream pose_data;
};

std::mutex current_session_mutex;
RecordingSession::Ptr current_session;

std::atomic<int> exposure_in_flight{0};
std::atomic<int> image_in_flight{0};
std::atomic<int> imu_in_flight{0};
std::atomic<int> pose_in_flight{0};
std::atomic<int> current_webp_quality{90};

tbb::concurrent_bounded_queue<std::array<float, 3>> imu_plot_queue;

#if CV_MAJOR_VERSION >= 3
std::string file_extension = ".webp";
#else
std::string file_extension = ".jpg";
#endif

// manual exposure mode, if not enabled will also record pose data
bool manual_exposure;

RecordingSession::Ptr getCurrentSession() {
  std::lock_guard<std::mutex> lock(current_session_mutex);
  return current_session;
}

void exposure_save_worker() {
  basalt::OpticalFlowInput::Ptr img;
  while (!stop_workers) {
    if (image_data_queue.try_pop(img)) {
      exposure_in_flight.fetch_add(1);
      auto session = getCurrentSession();
      for (size_t cam_id = 0; cam_id < NUM_CAMS; ++cam_id) {
        if (session) {
          session->cam_data[cam_id] << img->t_ns << "," << img->t_ns
                                    << file_extension << std::endl;

          session->exposure_data[cam_id]
              << img->t_ns << ","
              << int64_t(img->img_data[cam_id].exposure * 1e9) << std::endl;
        }
      }

      image_data_queue2.push(img);
      exposure_in_flight.fetch_sub(1);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void image_save_worker() {
  basalt::OpticalFlowInput::Ptr img;

  while (!stop_workers) {
    if (image_data_queue2.try_pop(img)) {
      image_in_flight.fetch_add(1);
      auto session = getCurrentSession();
      for (size_t cam_id = 0; cam_id < NUM_CAMS; ++cam_id) {
        basalt::ManagedImage<uint16_t>::Ptr image_raw =
            img->img_data[cam_id].img;

        if (!image_raw.get()) continue;

        cv::Mat image(image_raw->h, image_raw->w, CV_8U);

        uint8_t* dst = image.ptr();
        const uint16_t* src = image_raw->ptr;

        for (size_t i = 0; i < image_raw->size(); i++) {
          dst[i] = (src[i] >> 8);
        }

        if (!session) continue;

#if CV_MAJOR_VERSION >= 3
        std::string filename = session->dataset_dir + "mav0/cam" +
                               std::to_string(cam_id) + "/data/" +
                               std::to_string(img->t_ns) + ".webp";

        std::vector<int> compression_params = {cv::IMWRITE_WEBP_QUALITY,
                                               current_webp_quality.load()};
        cv::imwrite(filename, image, compression_params);
#else
        std::string filename = session->dataset_dir + "mav0/cam" +
                               std::to_string(cam_id) + "/data/" +
                               std::to_string(img->t_ns) + ".jpg";

        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY,
                                               current_webp_quality.load()};
        cv::imwrite(filename, image, compression_params);
#endif
      }
      image_in_flight.fetch_sub(1);

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void imu_save_worker() {
  basalt::ImuData<double>::Ptr data;

  while (!stop_workers) {
    if (imu_data_queue.try_pop(data)) {
      imu_in_flight.fetch_add(1);
      imu_plot_queue.try_push({static_cast<float>(data->accel[0]),
                               static_cast<float>(data->accel[1]),
                               static_cast<float>(data->accel[2])});

      auto session = getCurrentSession();
      if (session) {
        session->imu0_data << data->t_ns << "," << data->gyro[0] << ","
                           << data->gyro[1] << "," << data->gyro[2] << ","
                           << data->accel[0] << "," << data->accel[1] << ","
                           << data->accel[2] << "\n";
      }
      imu_in_flight.fetch_sub(1);

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void pose_save_worker() {
  basalt::RsPoseData data;

  while (!stop_workers) {
    if (pose_data_queue.try_pop(data)) {
      pose_in_flight.fetch_add(1);
      auto session = getCurrentSession();
      if (session) {
        session->pose_data << data.t_ns << "," << data.data.translation().x()
                           << "," << data.data.translation().y() << ","
                           << data.data.translation().z() << ","
                           << data.data.unit_quaternion().w() << ","
                           << data.data.unit_quaternion().x() << ","
                           << data.data.unit_quaternion().y() << ","
                           << data.data.unit_quaternion().z() << std::endl;
      }
      pose_in_flight.fetch_sub(1);

    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void save_calibration(const basalt::RsT265Device::Ptr& device,
                      const std::string& output_dir) {
  if (output_dir.empty()) return;

  auto calib = device->exportCalibration();

  if (calib) {
    std::ofstream os(output_dir + "/calibration.json");
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

void startRecording(const std::string& dir_path) {
  if (!recording) {
    if (stop_recording_thread.joinable()) stop_recording_thread.join();

    auto session = std::make_shared<RecordingSession>();
    session->dataset_dir = dir_path + "dataset_" + get_date() + "/";

    basalt::fs::create_directory(session->dataset_dir);
    basalt::fs::create_directory(session->dataset_dir + "mav0/");
    basalt::fs::create_directory(session->dataset_dir + "mav0/cam0/");
    basalt::fs::create_directory(session->dataset_dir + "mav0/cam0/data/");
    basalt::fs::create_directory(session->dataset_dir + "mav0/cam1/");
    basalt::fs::create_directory(session->dataset_dir + "mav0/cam1/data/");
    basalt::fs::create_directory(session->dataset_dir + "mav0/imu0/");

    session->cam_data[0].open(session->dataset_dir + "mav0/cam0/data.csv");
    session->cam_data[1].open(session->dataset_dir + "mav0/cam1/data.csv");
    session->exposure_data[0].open(session->dataset_dir +
                                   "mav0/cam0/exposure.csv");
    session->exposure_data[1].open(session->dataset_dir +
                                   "mav0/cam1/exposure.csv");
    session->imu0_data.open(session->dataset_dir + "mav0/imu0/data.csv");

    if (!manual_exposure) {
      basalt::fs::create_directory(session->dataset_dir + "mav0/realsense0/");
      session->pose_data.open(session->dataset_dir +
                              "mav0/realsense0/data.csv");
      session->pose_data
          << "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
             "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n";
    }

    session->cam_data[0] << "#timestamp [ns], filename\n";
    session->cam_data[1] << "#timestamp [ns], filename\n";
    session->exposure_data[0] << "#timestamp [ns], exposure time[ns]\n";
    session->exposure_data[1] << "#timestamp [ns], exposure time[ns]\n";
    session->imu0_data
        << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad "
           "s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y "
           "[m s^-2],a_RS_S_z [m s^-2]\n";

    {
      std::lock_guard<std::mutex> lock(current_session_mutex);
      current_session = session;
    }

    save_calibration(t265_device, session->dataset_dir);
    t265_device->setOutputQueues(&image_data_queue, &imu_data_queue,
                                 &pose_data_queue);

    std::cout << "Started recording dataset in " << session->dataset_dir
              << std::endl;

    recording = true;
  } else {
    std::cout << "Already recording" << std::endl;
  }
}

void stopRecording() {
  if (recording.exchange(false)) {
    auto session = getCurrentSession();
    auto stop_recording_func = [session]() {
      t265_device->detachOutputQueues();

      while (!image_data_queue.empty() || !image_data_queue2.empty() ||
             !imu_data_queue.empty() || !pose_data_queue.empty() ||
             exposure_in_flight.load() > 0 || image_in_flight.load() > 0 ||
             imu_in_flight.load() > 0 || pose_in_flight.load() > 0) {
        std::cout << "Waiting until the data from the queues is written to the "
                     "hard drive."
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }

      {
        std::lock_guard<std::mutex> lock(current_session_mutex);
        if (current_session == session) current_session.reset();
      }

      if (session) {
        std::cout << "Stopped recording dataset in " << session->dataset_dir
                  << std::endl;
      }
    };

    stop_recording_thread = std::thread(stop_recording_func);
  }
}

void toggleRecording(const std::string& dir_path) {
  if (recording) {
    stopRecording();
  } else {
    startRecording(dir_path);
  }
}

int main(int argc, char* argv[]) {
  CLI::App app{"Record RealSense T265 Data"};

  std::string dataset_path;

  app.add_option("--dataset-path", dataset_path, "Path to dataset");
  app.add_flag("--manual-exposure", manual_exposure,
               "If set will enable manual exposure.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (!dataset_path.empty() && dataset_path[dataset_path.length() - 1] != '/') {
    dataset_path += '/';
  }

  bool show_gui = true;

  stop_workers = false;
  recording = false;

  image_data_queue.set_capacity(1000);
  image_data_queue2.set_capacity(1000);
  imu_data_queue.set_capacity(10000);
  pose_data_queue.set_capacity(10000);
  imu_plot_queue.set_capacity(10000);

  if (worker_threads.empty()) {
    for (int i = 0; i < NUM_WORKERS; i++) {
      worker_threads.emplace_back(image_save_worker);
    }
  }

  exposure_save_thread = std::thread(exposure_save_worker);
  imu_worker_thread = std::thread(imu_save_worker);
  pose_worker_thread = std::thread(pose_save_worker);

  // realsense
  try {
    t265_device.reset(new basalt::RsT265Device(
        manual_exposure, skip_frames, current_webp_quality.load(), exposure));
    t265_device->start();
  } catch (const rs2::error& e) {
    std::cerr << "Failed to start RealSense T265: " << e.what() << std::endl;
    if (basalt::isUbuntu()) {
      basalt::printUbuntuUdevSetupInstructions(std::cerr);
    }
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Failed to start RealSense T265: " << e.what() << std::endl;
    if (basalt::isUbuntu()) {
      basalt::printUbuntuUdevSetupInstructions(std::cerr);
    }
    return 1;
  }

  imu_log.reset(new pangolin::DataLog);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Record RealSense T265", 1200, 800);

    pangolin::Var<std::function<void(void)>> record_btn(
        "ui.record", [&] { return toggleRecording(dataset_path); });
    pangolin::Var<std::function<void(void)>> export_calibration(
        "ui.export_calib", [&] {
          auto session = getCurrentSession();
          if (session) save_calibration(t265_device, session->dataset_dir);
        });

    std::atomic<int64_t> record_t_ns;
    record_t_ns = 0;

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < basalt::RsT265Device::NUM_CAMS) {
      int idx = img_view.size();
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      iv->extern_draw_function = [&, idx](pangolin::View& v) {
        UNUSED(v);

        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);  // red
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        auto last_img_data = t265_device->getLastImageData();

        if (last_img_data.get())
          pangolin::default_font()
              .Text("Exposure: %.3f ms.",
                    last_img_data->img_data[idx].exposure * 1000.0)
              .Draw(30, 30);

        if (idx == 0) {
          pangolin::default_font()
              .Text("Queue: %d.", image_data_queue2.size())
              .Draw(30, 60);
        }

        if (idx == 0 && recording) {
          pangolin::default_font().Text("Recording").Draw(30, 90);
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
      std::array<float, 3> imu_sample;
      while (imu_plot_queue.try_pop(imu_sample)) {
        imu_log->Log(imu_sample[0], imu_sample[1], imu_sample[2]);
      }

      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_SHORT;
        fmt.scalable_internal_format = GL_LUMINANCE16;

        auto last_img_data = t265_device->getLastImageData();
        if (last_img_data.get())
          for (size_t cam_id = 0; cam_id < basalt::RsT265Device::NUM_CAMS;
               cam_id++) {
            if (last_img_data->img_data[cam_id].img.get())
              img_view[cam_id]->SetImage(
                  last_img_data->img_data[cam_id].img->ptr,
                  last_img_data->img_data[cam_id].img->w,
                  last_img_data->img_data[cam_id].img->h,
                  last_img_data->img_data[cam_id].img->pitch, fmt);
          }
      }

      if (manual_exposure && exposure.GuiChanged()) {
        t265_device->setExposure(exposure);
      }

      if (webp_quality.GuiChanged()) {
        current_webp_quality = static_cast<int>(webp_quality);
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
  t265_device->stop();
  if (stop_recording_thread.joinable()) stop_recording_thread.join();
  stop_workers = true;

  for (auto& t : worker_threads) t.join();
  exposure_save_thread.join();
  imu_worker_thread.join();
  pose_worker_thread.join();

  return EXIT_SUCCESS;
}
