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

#include <CLI/CLI.hpp>

#include <basalt/utils/sophus_utils.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>

namespace cereal {

template <class Archive, class T, class C, class A>
inline void save(Archive& ar, std::map<std::string, T, C, A> const& map) {
  for (const auto& i : map) ar(cereal::make_nvp(i.first, i.second));
}

template <class Archive, class T, class C, class A>
inline void load(Archive& ar, std::map<std::string, T, C, A>& map) {
  map.clear();

  auto hint = map.begin();
  while (true) {
    const auto namePtr = ar.getNodeName();

    if (!namePtr) break;

    std::string key = namePtr;
    T value;
    ar(value);
    hint = map.emplace_hint(hint, std::move(key), std::move(value));
  }
}

}  // namespace cereal

Eigen::aligned_vector<Sophus::SE3d> load_poses(const std::string& path) {
  Eigen::aligned_vector<Sophus::SE3d> res;

  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line[0] == '#') continue;

    std::stringstream ss(line);

    Eigen::Matrix3d rot;
    Eigen::Vector3d pos;

    ss >> rot(0, 0) >> rot(0, 1) >> rot(0, 2) >> pos[0] >> rot(1, 0) >>
        rot(1, 1) >> rot(1, 2) >> pos[1] >> rot(2, 0) >> rot(2, 1) >>
        rot(2, 2) >> pos[2];

    res.emplace_back(Eigen::Quaterniond(rot), pos);
  }

  return res;
}

void eval_kitti(const std::vector<double>& lengths,
                const Eigen::aligned_vector<Sophus::SE3d>& poses_gt,
                const Eigen::aligned_vector<Sophus::SE3d>& poses_result,
                std::map<std::string, std::map<std::string, double>>& res) {
  auto lastFrameFromSegmentLength = [](std::vector<float>& dist,
                                       int first_frame, float len) {
    for (int i = first_frame; i < (int)dist.size(); i++)
      if (dist[i] > dist[first_frame] + len) return i;
    return -1;
  };

  std::cout << "poses_gt.size() " << poses_gt.size() << std::endl;
  std::cout << "poses_result.size() " << poses_result.size() << std::endl;

  // pre-compute distances (from ground truth as reference)
  std::vector<float> dist_gt;
  dist_gt.emplace_back(0);
  for (size_t i = 1; i < poses_gt.size(); i++) {
    const auto& p1 = poses_gt[i - 1];
    const auto& p2 = poses_gt[i];

    dist_gt.emplace_back(dist_gt.back() +
                         (p2.translation() - p1.translation()).norm());
  }

  const size_t step_size = 10;

  for (size_t i = 0; i < lengths.size(); i++) {
    // current length
    float len = lengths[i];

    double t_error_sum = 0;
    double r_error_sum = 0;
    int num_meas = 0;

    for (size_t first_frame = 0; first_frame < poses_gt.size();
         first_frame += step_size) {
      // for all segment lengths do

      // compute last frame
      int32_t last_frame =
          lastFrameFromSegmentLength(dist_gt, first_frame, len);

      // continue, if sequence not long enough
      if (last_frame == -1) continue;

      // compute rotational and translational errors
      Sophus::SE3d pose_delta_gt =
          poses_gt[first_frame].inverse() * poses_gt[last_frame];
      Sophus::SE3d pose_delta_result =
          poses_result[first_frame].inverse() * poses_result[last_frame];
      // Sophus::SE3d pose_error = pose_delta_result.inverse() * pose_delta_gt;
      double r_err = pose_delta_result.unit_quaternion().angularDistance(
                         pose_delta_gt.unit_quaternion()) *
                     180.0 / M_PI;
      double t_err =
          (pose_delta_result.translation() - pose_delta_gt.translation())
              .norm();

      t_error_sum += t_err / len;
      r_error_sum += r_err / len;
      num_meas++;
    }

    std::string len_str = std::to_string((int)len);
    res[len_str]["trans_error"] = 100.0 * t_error_sum / num_meas;
    res[len_str]["rot_error"] = r_error_sum / num_meas;
    res[len_str]["num_meas"] = num_meas;
  }
}

int main(int argc, char** argv) {
  std::vector<double> lengths = {100, 200, 300, 400, 500, 600, 700, 800};
  std::string result_path;
  std::string traj_path;
  std::string gt_path;

  CLI::App app{"KITTI evaluation"};

  app.add_option("--traj-path", traj_path,
                 "Path to the file with computed trajectory.")
      ->required();
  app.add_option("--gt-path", gt_path,
                 "Path to the file with ground truth trajectory.")
      ->required();
  app.add_option("--result-path", result_path, "Path to store the result file.")
      ->required();

  app.add_option("--eval-lengths", lengths, "Trajectory length to evaluate.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  const Eigen::aligned_vector<Sophus::SE3d> poses_gt = load_poses(gt_path);
  const Eigen::aligned_vector<Sophus::SE3d> poses_result =
      load_poses(traj_path);

  if (poses_gt.empty() || poses_gt.size() != poses_result.size()) {
    std::cerr << "Wrong number of poses: poses_gt " << poses_gt.size()
              << " poses_result " << poses_result.size() << std::endl;
    std::abort();
  }

  std::map<std::string, std::map<std::string, double>> res_map;
  eval_kitti(lengths, poses_gt, poses_result, res_map);

  {
    cereal::JSONOutputArchive ar(std::cout);
    ar(cereal::make_nvp("results", res_map));
    std::cout << std::endl;
  }

  if (!result_path.empty()) {
    std::ofstream os(result_path);
    {
      cereal::JSONOutputArchive ar(os);
      ar(cereal::make_nvp("results", res_map));
    }
    os.close();
  }
}
