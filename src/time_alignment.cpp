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

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <basalt/io/dataset_io.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/filesystem.h>
#include <basalt/calibration/calibration.hpp>

#include <CLI/CLI.hpp>

basalt::Calibration<double> calib;
basalt::MocapCalibration<double> mocap_calib;

// Linear time version
double compute_error(
    int64_t offset, const std::vector<int64_t> &gyro_timestamps,
    const Eigen::aligned_vector<Eigen::Vector3d> &gyro_data,
    const std::vector<int64_t> &mocap_rot_vel_timestamps,
    const Eigen::aligned_vector<Eigen::Vector3d> &mocap_rot_vel_data) {
  double error = 0;
  int num_points = 0;

  size_t j = 0;

  for (size_t i = 0; i < mocap_rot_vel_timestamps.size(); i++) {
    int64_t corrected_time = mocap_rot_vel_timestamps[i] + offset;

    while (gyro_timestamps[j] < corrected_time) j++;
    if (j >= gyro_timestamps.size()) break;

    int64_t dist_j = gyro_timestamps[j] - corrected_time;
    int64_t dist_j_m1 = corrected_time - gyro_timestamps[j - 1];

    BASALT_ASSERT(dist_j >= 0);
    BASALT_ASSERT(dist_j_m1 >= 0);

    int idx = dist_j < dist_j_m1 ? j : j - 1;

    if (std::min(dist_j, dist_j_m1) > 1e9 / 120) continue;

    error += (gyro_data[idx] - mocap_rot_vel_data[i]).norm();
    num_points++;
  }
  return error / num_points;
}

int main(int argc, char **argv) {
  std::string dataset_path;
  std::string calibration_path;
  std::string mocap_calibration_path;
  std::string dataset_type;
  std::string output_path;
  std::string output_error_path;
  std::string output_gyro_path;
  std::string output_mocap_path;

  double max_offset_s = 10.0;

  bool show_gui = true;

  CLI::App app{"Calibrate time offset"};

  app.add_option("-d,--dataset-path", dataset_path, "Path to dataset")
      ->required();
  app.add_option("--calibration", calibration_path, "Path to calibration file");
  app.add_option("--mocap-calibration", mocap_calibration_path,
                 "Path to mocap calibration file");
  app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")
      ->required();

  app.add_option("--output", output_path,
                 "Path to output file with time-offset result");
  app.add_option("--output-error", output_error_path,
                 "Path to output file with error time-series for plotting");
  app.add_option(
      "--output-gyro", output_gyro_path,
      "Path to output file with gyro rotational velocities for plotting");
  app.add_option(
      "--output-mocap", output_mocap_path,
      "Path to output file with mocap rotational velocities for plotting");

  app.add_option("--max-offset", max_offset_s,
                 "Maximum offset for a grid search in seconds.");

  app.add_flag("--show-gui", show_gui, "Show GUI for debugging");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (!dataset_path.empty() && dataset_path[dataset_path.length() - 1] != '/') {
    dataset_path += '/';
  }

  basalt::VioDatasetPtr vio_dataset;

  const bool use_calib =
      !(calibration_path.empty() || mocap_calibration_path.empty());

  if (use_calib) {
    std::ifstream is(calibration_path);

    if (is.good()) {
      cereal::JSONInputArchive archive(is);
      archive(calib);
      std::cout << "Loaded calibration from: " << calibration_path << std::endl;
    } else {
      std::cerr << "No calibration found" << std::endl;
      std::abort();
    }

    std::ifstream mocap_is(mocap_calibration_path);

    if (mocap_is.good()) {
      cereal::JSONInputArchive archive(mocap_is);
      archive(mocap_calib);
      std::cout << "Loaded mocap calibration from: " << mocap_calibration_path
                << std::endl;
    } else {
      std::cerr << "No mocap calibration found" << std::endl;
      std::abort();
    }
  }

  basalt::DatasetIoInterfacePtr dataset_io =
      basalt::DatasetIoFactory::getDatasetIo(dataset_type, true);

  dataset_io->read(dataset_path);
  vio_dataset = dataset_io->get_data();

  std::vector<int64_t> gyro_timestamps;
  Eigen::aligned_vector<Eigen::Vector3d> gyro_data;

  std::vector<int64_t> mocap_rot_vel_timestamps;
  Eigen::aligned_vector<Eigen::Vector3d> mocap_rot_vel_data;

  // Apply calibration to gyro
  {
    int saturation_count = 0;
    for (size_t i = 0; i < vio_dataset->get_gyro_data().size(); i++) {
      if (vio_dataset->get_gyro_data()[i].data.array().abs().maxCoeff() >
          499.0 * M_PI / 180) {
        ++saturation_count;
        continue;
      }
      gyro_timestamps.push_back(vio_dataset->get_gyro_data()[i].timestamp_ns);

      Eigen::Vector3d measurement = vio_dataset->get_gyro_data()[i].data;
      if (use_calib) {
        gyro_data.push_back(calib.calib_gyro_bias.getCalibrated(measurement));
      } else {
        gyro_data.push_back(measurement);
      }
    }
    std::cout << "saturated gyro measurement count: " << saturation_count
              << std::endl;
  }

  // compute rotational velocity from mocap data
  {
    Sophus::SE3d T_mark_i;
    if (use_calib) T_mark_i = mocap_calib.T_i_mark.inverse();

    int saturation_count = 0;
    for (size_t i = 1; i < vio_dataset->get_gt_timestamps().size() - 1; i++) {
      Sophus::SE3d p0, p1;

      // compute central differences, to have no timestamp bias
      p0 = vio_dataset->get_gt_pose_data()[i - 1] * T_mark_i;
      p1 = vio_dataset->get_gt_pose_data()[i + 1] * T_mark_i;

      double dt = (vio_dataset->get_gt_timestamps()[i + 1] -
                   vio_dataset->get_gt_timestamps()[i - 1]) *
                  1e-9;

      // only compute difference, if measurements are really 2 consecutive
      // measurements apart (assuming 120 Hz data)
      if (dt > 2.5 / 120) continue;

      Eigen::Vector3d rot_vel = (p0.so3().inverse() * p1.so3()).log() / dt;

      // Filter outliers
      if (rot_vel.array().abs().maxCoeff() > 500 * M_PI / 180) {
        ++saturation_count;
        continue;
      }

      mocap_rot_vel_timestamps.push_back(vio_dataset->get_gt_timestamps()[i]);
      mocap_rot_vel_data.push_back(rot_vel);
    }
    std::cout << "outlier mocap rotation velocity count: " << saturation_count
              << std::endl;
  }

  std::cout << "gyro_data.size() " << gyro_data.size() << std::endl;
  std::cout << "mocap_rot_vel_data.size() " << mocap_rot_vel_data.size()
            << std::endl;

  std::vector<double> offsets_vec;
  std::vector<double> errors_vec;

  int best_offset_ns = 0;
  double best_error = std::numeric_limits<double>::max();
  int best_error_idx = -1;

  int64_t max_offset_ns = max_offset_s * 1e9;
  int64_t offset_inc_ns = 100000;

  for (int64_t offset_ns = -max_offset_ns; offset_ns <= max_offset_ns;
       offset_ns += offset_inc_ns) {
    double error = compute_error(offset_ns, gyro_timestamps, gyro_data,
                                 mocap_rot_vel_timestamps, mocap_rot_vel_data);

    offsets_vec.push_back(offset_ns * 1e-6);
    errors_vec.push_back(error);

    if (error < best_error) {
      best_error = error;
      best_offset_ns = offset_ns;
      best_error_idx = errors_vec.size() - 1;
    }
  }

  std::cout << "Best error: " << best_error << std::endl;
  std::cout << "Best error idx : " << best_error_idx << std::endl;
  std::cout << "Best offset: " << best_offset_ns << std::endl;

  pangolin::DataLog error_log;

  int best_offset_refined_ns = best_offset_ns;

  // Subpixel accuracy
  Eigen::Vector3d coeff(0, 0, 0);
  {
    const static int SAMPLE_INTERVAL = 10;

    if (best_error_idx - SAMPLE_INTERVAL >= 0 &&
        best_error_idx + SAMPLE_INTERVAL < int(errors_vec.size())) {
      Eigen::MatrixXd pol(2 * SAMPLE_INTERVAL + 1, 3);
      Eigen::VectorXd err(2 * SAMPLE_INTERVAL + 1);

      for (int i = 0; i < 2 * SAMPLE_INTERVAL + 1; i++) {
        int idx = i - SAMPLE_INTERVAL;
        pol(i, 0) = idx * idx;
        pol(i, 1) = idx;
        pol(i, 2) = 1;

        err(i) = errors_vec[best_error_idx + idx];
      }

      coeff =
          pol.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(err);

      double a = coeff[0];
      double b = coeff[1];

      if (a > 1e-9) {
        best_offset_refined_ns -= offset_inc_ns * b / (2 * a);
      }
    }

    for (size_t i = 0; i < errors_vec.size(); i++) {
      const double idx =
          static_cast<double>(static_cast<int>(i) - best_error_idx);

      const Eigen::Vector3d pol(idx * idx, idx, 1);

      error_log.Log(offsets_vec[i], errors_vec[i], pol.transpose() * coeff);
    }
  }

  std::cout << "Best error refined: "
            << compute_error(best_offset_refined_ns, gyro_timestamps, gyro_data,
                             mocap_rot_vel_timestamps, mocap_rot_vel_data)
            << std::endl;
  std::cout << "Best offset refined: " << best_offset_refined_ns << std::endl;

  std::cout << "Total mocap offset: "
            << vio_dataset->get_mocap_to_imu_offset_ns() +
                   best_offset_refined_ns
            << std::endl;

  if (output_path != "") {
    std::ofstream os(output_path);
    cereal::JSONOutputArchive archive(os);
    archive(cereal::make_nvp("mocap_to_imu_initial_offset_ns",
                             vio_dataset->get_mocap_to_imu_offset_ns()));
    archive(cereal::make_nvp("mocap_to_imu_additional_offset_refined_ns",
                             best_offset_refined_ns));
    archive(cereal::make_nvp(
        "mocap_to_imu_total_offset_ns",
        vio_dataset->get_mocap_to_imu_offset_ns() + best_offset_refined_ns));
  }

  if (output_error_path != "") {
    std::cout << "Writing error time series to '" << output_error_path << "'"
              << std::endl;

    std::ofstream os(output_error_path);
    os << "#TIME_MS,ERROR,ERROR_FITTED" << std::endl;
    os << "# best_offset_ms: " << best_offset_ns * 1e-6
       << ", best_offset_refined_ms: " << best_offset_refined_ns * 1e-6
       << std::endl;

    for (size_t i = 0; i < errors_vec.size(); ++i) {
      const double idx =
          static_cast<double>(static_cast<int>(i) - best_error_idx);
      const Eigen::Vector3d pol(idx * idx, idx, 1);
      const double fitted = pol.transpose() * coeff;
      os << offsets_vec[i] << "," << errors_vec[i] << "," << fitted
         << std::endl;
    }
  }

  const int64_t min_time = vio_dataset->get_gyro_data().front().timestamp_ns;
  const int64_t max_time = vio_dataset->get_gyro_data().back().timestamp_ns;

  if (output_gyro_path != "") {
    std::cout << "Writing gyro values to '" << output_gyro_path << "'"
              << std::endl;

    std::ofstream os(output_gyro_path);
    os << "#TIME_M, GX, GY, GZ" << std::endl;

    for (size_t i = 0; i < gyro_timestamps.size(); ++i) {
      os << (gyro_timestamps[i] - min_time) * 1e-9 << " "
         << gyro_data[i].transpose() << std::endl;
    }
  }

  if (output_mocap_path != "") {
    std::cout << "Writing mocap rotational velocity values to '"
              << output_mocap_path << "'" << std::endl;

    std::ofstream os(output_mocap_path);
    os << "#TIME_M, GX, GY, GZ" << std::endl;

    for (size_t i = 0; i < gyro_timestamps.size(); ++i) {
      os << (mocap_rot_vel_timestamps[i] + best_offset_ns - min_time) * 1e-9
         << " " << mocap_rot_vel_data[i].transpose() << std::endl;
    }
  }

  if (show_gui) {
    static constexpr int UI_WIDTH = 280;

    pangolin::CreateWindowAndBind("Main", 1280, 800);

    pangolin::Plotter *plotter;

    pangolin::DataLog data_log, mocap_log;

    pangolin::View &plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    plotter = new pangolin::Plotter(&data_log, 0, (max_time - min_time) * 1e-9,
                                    -10.0, 10.0, 0.01, 0.01);

    plot_display.AddDisplay(*plotter);
    pangolin::Var<bool> show_gyro("ui.show_gyro", true, false, true);
    pangolin::Var<bool> show_mocap_rot_vel("ui.show_mocap_rot_vel", true, false,
                                           true);

    pangolin::Var<bool> show_error("ui.show_error", false, false, true);

    std::string save_button_name = "ui.save_aligned_dataset";
    // Disable save_aligned_dataset button if GT data already exists
    if (basalt::fs::exists(
            basalt::fs::path(dataset_path + "mav0/gt/data.csv"))) {
      save_button_name += "(disabled)";
    }

    pangolin::Var<std::function<void(void)>> save_aligned_dataset(
        save_button_name, [&]() {
          if (basalt::fs::exists(
                  basalt::fs::path(dataset_path + "mav0/gt/data.csv"))) {
            std::cout << "Aligned ground-truth data already exists, skipping. "
                         "If you want to run the calibration again delete "
                      << dataset_path << "mav0/gt/ folder." << std::endl;
            return;
          }
          std::cout << "Saving aligned dataset in "
                    << dataset_path + "mav0/gt/data.csv" << std::endl;
          // output corrected mocap data
          Sophus::SE3d T_mark_i;
          if (use_calib) T_mark_i = mocap_calib.T_i_mark.inverse();
          basalt::fs::create_directory(dataset_path + "mav0/gt/");
          std::ofstream gt_out_stream;
          gt_out_stream.open(dataset_path + "mav0/gt/data.csv");
          gt_out_stream
              << "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
                 "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n";

          for (size_t i = 0; i < vio_dataset->get_gt_timestamps().size(); i++) {
            gt_out_stream << vio_dataset->get_gt_timestamps()[i] +
                                 best_offset_refined_ns
                          << ",";
            Sophus::SE3d pose_corrected =
                vio_dataset->get_gt_pose_data()[i] * T_mark_i;
            gt_out_stream << pose_corrected.translation().x() << ","
                          << pose_corrected.translation().y() << ","
                          << pose_corrected.translation().z() << ","
                          << pose_corrected.unit_quaternion().w() << ","
                          << pose_corrected.unit_quaternion().x() << ","
                          << pose_corrected.unit_quaternion().y() << ","
                          << pose_corrected.unit_quaternion().z() << std::endl;
          }
          gt_out_stream.close();
        });

    auto recompute_logs = [&]() {
      data_log.Clear();
      mocap_log.Clear();

      for (size_t i = 0; i < gyro_timestamps.size(); i++) {
        data_log.Log((gyro_timestamps[i] - min_time) * 1e-9, gyro_data[i][0],
                     gyro_data[i][1], gyro_data[i][2]);
      }

      for (size_t i = 0; i < mocap_rot_vel_timestamps.size(); i++) {
        mocap_log.Log(
            (mocap_rot_vel_timestamps[i] + best_offset_ns - min_time) * 1e-9,
            mocap_rot_vel_data[i][0], mocap_rot_vel_data[i][1],
            mocap_rot_vel_data[i][2]);
      }
    };

    auto drawPlots = [&]() {
      plotter->ClearSeries();
      plotter->ClearMarkers();

      if (show_gyro) {
        plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                           pangolin::Colour::Red(), "g x");
        plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                           pangolin::Colour::Green(), "g y");
        plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                           pangolin::Colour::Blue(), "g z");
      }

      if (show_mocap_rot_vel) {
        plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                           pangolin::Colour(1, 1, 0), "pv x", &mocap_log);
        plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                           pangolin::Colour(1, 0, 1), "pv y", &mocap_log);
        plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                           pangolin::Colour(0, 1, 1), "pv z", &mocap_log);
      }

      if (show_error) {
        plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                           pangolin::Colour(1, 1, 1), "error", &error_log);
        plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                           pangolin::Colour(0.3, 1, 0.8), "fitted error",
                           &error_log);
        plotter->AddMarker(pangolin::Marker::Vertical,
                           best_offset_refined_ns * 1e-6,
                           pangolin::Marker::Equal, pangolin::Colour(1, 0, 0));
      }
    };

    recompute_logs();
    drawPlots();

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (show_gyro.GuiChanged() || show_mocap_rot_vel.GuiChanged() ||
          show_error.GuiChanged()) {
        drawPlots();
      }

      pangolin::FinishFrame();
    }
  }

  return 0;
}
