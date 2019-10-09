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

#include <basalt/io/dataset_io.h>
#include <basalt/utils/filesystem.h>

#include <opencv2/highgui/highgui.hpp>

namespace basalt {

class UzhVioDataset : public VioDataset {
  size_t num_cams;

  std::string path;

  std::vector<int64_t> image_timestamps;
  std::unordered_map<int64_t, std::string> left_image_path;
  std::unordered_map<int64_t, std::string> right_image_path;

  // vector of images for every timestamp
  // assumes vectors size is num_cams for every timestamp with null pointers for
  // missing frames
  // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
  Eigen::aligned_vector<Sophus::SE3d>
      gt_pose_data;  // TODO: change to eigen aligned

  int64_t mocap_to_imu_offset_ns = 0;

  std::vector<std::unordered_map<int64_t, double>> exposure_times;

 public:
  ~UzhVioDataset(){};

  size_t get_num_cams() const { return num_cams; }

  std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

  const Eigen::aligned_vector<AccelData> &get_accel_data() const {
    return accel_data;
  }
  const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
    return gyro_data;
  }
  const std::vector<int64_t> &get_gt_timestamps() const {
    return gt_timestamps;
  }
  const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
    return gt_pose_data;
  }

  int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) {
    std::vector<ImageData> res(num_cams);

    for (size_t i = 0; i < num_cams; i++) {
      std::string full_image_path =
          path + "/" +
          (i == 0 ? left_image_path.at(t_ns) : right_image_path.at(t_ns));

      if (fs::exists(full_image_path)) {
        cv::Mat img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);

        if (img.type() == CV_8UC1) {
          res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));

          const uint8_t *data_in = img.ptr();
          uint16_t *data_out = res[i].img->ptr;

          size_t full_size = img.cols * img.rows;
          for (size_t i = 0; i < full_size; i++) {
            int val = data_in[i];
            val = val << 8;
            data_out[i] = val;
          }
        } else if (img.type() == CV_8UC3) {
          res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));

          const uint8_t *data_in = img.ptr();
          uint16_t *data_out = res[i].img->ptr;

          size_t full_size = img.cols * img.rows;
          for (size_t i = 0; i < full_size; i++) {
            int val = data_in[i * 3];
            val = val << 8;
            data_out[i] = val;
          }
        } else if (img.type() == CV_16UC1) {
          res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));
          std::memcpy(res[i].img->ptr, img.ptr(),
                      img.cols * img.rows * sizeof(uint16_t));

        } else {
          std::cerr << "img.fmt.bpp " << img.type() << std::endl;
          std::abort();
        }

        auto exp_it = exposure_times[i].find(t_ns);
        if (exp_it != exposure_times[i].end()) {
          res[i].exposure = exp_it->second;
        }
      }
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class UzhIO;
};

class UzhIO : public DatasetIoInterface {
 public:
  UzhIO() {}

  void read(const std::string &path) {
    if (!fs::exists(path))
      std::cerr << "No dataset found in " << path << std::endl;

    data.reset(new UzhVioDataset);

    data->num_cams = 2;
    data->path = path;

    read_image_timestamps(path);

    std::cout << "Loaded " << data->get_image_timestamps().size()
              << " timestamps, " << data->left_image_path.size()
              << " left images and " << data->right_image_path.size()
              << std::endl;

    //    {
    //      int64_t t_ns = data->get_image_timestamps()[0];
    //      std::cout << t_ns << " " << data->left_image_path.at(t_ns) << " "
    //                << data->right_image_path.at(t_ns) << std::endl;
    //    }

    read_imu_data(path + "/imu.txt");

    std::cout << "Loaded " << data->get_gyro_data().size() << " imu msgs."
              << std::endl;

    if (fs::exists(path + "/groundtruth.txt")) {
      read_gt_data_pose(path + "/groundtruth.txt");
    }

    data->exposure_times.resize(data->num_cams);
  }

  void reset() { data.reset(); }

  VioDatasetPtr get_data() { return data; }

 private:
  void read_exposure(const std::string &path,
                     std::unordered_map<int64_t, double> &exposure_data) {
    exposure_data.clear();

    std::ifstream f(path + "exposure.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      int64_t timestamp, exposure_int;
      Eigen::Vector3d gyro, accel;

      ss >> timestamp >> tmp >> exposure_int;

      exposure_data[timestamp] = exposure_int * 1e-9;
    }
  }

  void read_image_timestamps(const std::string &path) {
    {
      std::ifstream f(path + "/left_images.txt");
      std::string line;
      while (std::getline(f, line)) {
        if (line[0] == '#') continue;
        std::stringstream ss(line);
        int tmp;
        double t_s;
        std::string path;
        ss >> tmp >> t_s >> path;

        int64_t t_ns = t_s * 1e9;

        data->image_timestamps.emplace_back(t_ns);
        data->left_image_path[t_ns] = path;
      }
    }

    {
      std::ifstream f(path + "/right_images.txt");
      std::string line;
      while (std::getline(f, line)) {
        if (line[0] == '#') continue;
        std::stringstream ss(line);
        int tmp;
        double t_s;
        std::string path;
        ss >> tmp >> t_s >> path;

        int64_t t_ns = t_s * 1e9;

        data->right_image_path[t_ns] = path;
      }
    }
  }

  void read_imu_data(const std::string &path) {
    data->accel_data.clear();
    data->gyro_data.clear();

    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      int tmp;
      double timestamp;
      Eigen::Vector3d gyro, accel;

      ss >> tmp >> timestamp >> gyro[0] >> gyro[1] >> gyro[2] >> accel[0] >>
          accel[1] >> accel[2];

      int64_t t_ns = timestamp * 1e9;

      data->accel_data.emplace_back();
      data->accel_data.back().timestamp_ns = t_ns;
      data->accel_data.back().data = accel;

      data->gyro_data.emplace_back();
      data->gyro_data.back().timestamp_ns = t_ns;
      data->gyro_data.back().data = gyro;
    }
  }

  void read_gt_data_pose(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      int tmp;
      double timestamp;
      Eigen::Quaterniond q;
      Eigen::Vector3d pos;

      ss >> tmp >> timestamp >> pos[0] >> pos[1] >> pos[2] >> q.x() >> q.y() >>
          q.z() >> q.w();

      int64_t t_ns = timestamp * 1e9;

      data->gt_timestamps.emplace_back(t_ns);
      data->gt_pose_data.emplace_back(q, pos);
    }
  }

  std::shared_ptr<UzhVioDataset> data;
};

}  // namespace basalt
