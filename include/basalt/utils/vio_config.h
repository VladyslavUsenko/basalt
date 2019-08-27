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

#include <string>

namespace basalt {

struct VioConfig {
  VioConfig();
  void load(const std::string& filename);
  void save(const std::string& filename);

  std::string optical_flow_type;
  int optical_flow_detection_grid_size;
  float optical_flow_max_recovered_dist2;
  int optical_flow_pattern;
  int optical_flow_max_iterations;
  int optical_flow_levels;
  float optical_flow_epipolar_error;
  int optical_flow_skip_frames;

  int vio_max_states;
  int vio_max_kfs;
  int vio_min_frames_after_kf;
  float vio_new_kf_keypoints_thresh;
  bool vio_debug;

  double vio_outlier_threshold;
  int vio_filter_iteration;
  int vio_max_iterations;

  double vio_obs_std_dev;
  double vio_obs_huber_thresh;
  double vio_min_triangulation_dist;

  bool vio_enforce_realtime;

  bool vio_use_lm;
  double vio_lm_lambda_min;
  double vio_lm_lambda_max;

  double vio_init_pose_weight;
  double vio_init_ba_weight;
  double vio_init_bg_weight;

  double mapper_obs_std_dev;
  double mapper_obs_huber_thresh;
  int mapper_detection_num_points;
  double mapper_num_frames_to_match;
  double mapper_frames_to_match_threshold;
  double mapper_min_matches;
  double mapper_ransac_threshold;
  double mapper_min_track_length;
  double mapper_max_hamming_distance;
  double mapper_second_best_test_ratio;
  int mapper_bow_num_bits;
  double mapper_min_triangulation_dist;
  bool mapper_no_factor_weights;
  bool mapper_use_factors;

  bool mapper_use_lm;
  double mapper_lm_lambda_min;
  double mapper_lm_lambda_max;
};
}  // namespace basalt
