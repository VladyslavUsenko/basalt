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

#include <basalt/utils/vio_config.h>

#include <fstream>

#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>

namespace basalt {

VioConfig::VioConfig() {
  // optical_flow_type = "patch";
  optical_flow_type = "frame_to_frame";
  optical_flow_detection_grid_size = 50;
  optical_flow_max_recovered_dist2 = 0.09f;
  optical_flow_pattern = 51;
  optical_flow_max_iterations = 5;
  optical_flow_levels = 3;
  optical_flow_epipolar_error = 0.005;
  optical_flow_skip_frames = 1;

  vio_max_states = 3;
  vio_max_kfs = 7;
  vio_min_frames_after_kf = 5;
  vio_new_kf_keypoints_thresh = 0.7;

  vio_debug = false;
  vio_obs_std_dev = 0.5;
  vio_obs_huber_thresh = 1.0;
  vio_min_triangulation_dist = 0.05;
  vio_outlier_threshold = 3.0;
  vio_filter_iteration = 4;
  vio_max_iterations = 7;

  vio_enforce_realtime = false;

  vio_use_lm = false;
  vio_lm_lambda_min = 1e-32;
  vio_lm_lambda_max = 1e2;

  vio_init_pose_weight = 1e8;
  vio_init_ba_weight = 1e1;
  vio_init_bg_weight = 1e2;

  mapper_obs_std_dev = 0.25;
  mapper_obs_huber_thresh = 1.5;
  mapper_detection_num_points = 800;
  mapper_num_frames_to_match = 30;
  mapper_frames_to_match_threshold = 0.04;
  mapper_min_matches = 20;
  mapper_ransac_threshold = 5e-5;
  mapper_min_track_length = 5;
  mapper_max_hamming_distance = 70;
  mapper_second_best_test_ratio = 1.2;
  mapper_bow_num_bits = 16;
  mapper_min_triangulation_dist = 0.07;
  mapper_no_factor_weights = false;
  mapper_use_factors = true;

  mapper_use_lm = false;
  mapper_lm_lambda_min = 1e-32;
  mapper_lm_lambda_max = 1e2;
}

void VioConfig::save(const std::string& filename) {
  std::ofstream os(filename);

  {
    cereal::JSONOutputArchive archive(os);
    archive(*this);
  }
  os.close();
}

void VioConfig::load(const std::string& filename) {
  std::ifstream is(filename);

  {
    cereal::JSONInputArchive archive(is);
    archive(*this);
  }
  is.close();
}
}  // namespace basalt

namespace cereal {

template <class Archive>
void serialize(Archive& ar, basalt::VioConfig& config) {
  ar(CEREAL_NVP(config.optical_flow_type));
  ar(CEREAL_NVP(config.optical_flow_detection_grid_size));
  ar(CEREAL_NVP(config.optical_flow_max_recovered_dist2));
  ar(CEREAL_NVP(config.optical_flow_pattern));
  ar(CEREAL_NVP(config.optical_flow_max_iterations));
  ar(CEREAL_NVP(config.optical_flow_epipolar_error));
  ar(CEREAL_NVP(config.optical_flow_levels));
  ar(CEREAL_NVP(config.optical_flow_skip_frames));

  ar(CEREAL_NVP(config.vio_max_states));
  ar(CEREAL_NVP(config.vio_max_kfs));
  ar(CEREAL_NVP(config.vio_min_frames_after_kf));
  ar(CEREAL_NVP(config.vio_new_kf_keypoints_thresh));
  ar(CEREAL_NVP(config.vio_debug));
  ar(CEREAL_NVP(config.vio_max_iterations));
  ar(CEREAL_NVP(config.vio_outlier_threshold));
  ar(CEREAL_NVP(config.vio_filter_iteration));

  ar(CEREAL_NVP(config.vio_obs_std_dev));
  ar(CEREAL_NVP(config.vio_obs_huber_thresh));
  ar(CEREAL_NVP(config.vio_min_triangulation_dist));

  ar(CEREAL_NVP(config.vio_enforce_realtime));

  ar(CEREAL_NVP(config.vio_use_lm));
  ar(CEREAL_NVP(config.vio_lm_lambda_min));
  ar(CEREAL_NVP(config.vio_lm_lambda_max));

  ar(CEREAL_NVP(config.vio_init_pose_weight));
  ar(CEREAL_NVP(config.vio_init_ba_weight));
  ar(CEREAL_NVP(config.vio_init_bg_weight));

  ar(CEREAL_NVP(config.mapper_obs_std_dev));
  ar(CEREAL_NVP(config.mapper_obs_huber_thresh));
  ar(CEREAL_NVP(config.mapper_detection_num_points));
  ar(CEREAL_NVP(config.mapper_num_frames_to_match));
  ar(CEREAL_NVP(config.mapper_frames_to_match_threshold));
  ar(CEREAL_NVP(config.mapper_min_matches));
  ar(CEREAL_NVP(config.mapper_ransac_threshold));
  ar(CEREAL_NVP(config.mapper_min_track_length));
  ar(CEREAL_NVP(config.mapper_max_hamming_distance));
  ar(CEREAL_NVP(config.mapper_second_best_test_ratio));
  ar(CEREAL_NVP(config.mapper_bow_num_bits));
  ar(CEREAL_NVP(config.mapper_min_triangulation_dist));
  ar(CEREAL_NVP(config.mapper_no_factor_weights));
  ar(CEREAL_NVP(config.mapper_use_factors));

  ar(CEREAL_NVP(config.mapper_use_lm));
  ar(CEREAL_NVP(config.mapper_lm_lambda_min));
  ar(CEREAL_NVP(config.mapper_lm_lambda_max));
}
}  // namespace cereal
