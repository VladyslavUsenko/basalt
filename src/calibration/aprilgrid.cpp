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

#include <fstream>

#include <basalt/calibration/aprilgrid.h>

#include <cereal/archives/json.hpp>

namespace basalt {

AprilGrid::AprilGrid(const std::string &config_path) {
  std::ifstream is(config_path);
  if (is.is_open()) {
    cereal::JSONInputArchive ar(is);
    ar(cereal::make_nvp("tagCols", tagCols));
    ar(cereal::make_nvp("tagRows", tagRows));
    ar(cereal::make_nvp("tagSize", tagSize));
    ar(cereal::make_nvp("tagSpacing", tagSpacing));
  } else {
    std::cerr << "Could not open aprilgrid configuration: " << config_path
              << std::endl;
    std::abort();
  }

  double x_corner_offsets[4] = {0, tagSize, tagSize, 0};
  double y_corner_offsets[4] = {0, 0, tagSize, tagSize};

  aprilgrid_corner_pos_3d.resize(tagCols * tagRows * 4);

  for (int y = 0; y < tagRows; y++) {
    for (int x = 0; x < tagCols; x++) {
      int tag_id = tagCols * y + x;
      double x_offset = x * tagSize * (1 + tagSpacing);
      double y_offset = y * tagSize * (1 + tagSpacing);

      for (int i = 0; i < 4; i++) {
        int corner_id = (tag_id << 2) + i;

        Eigen::Vector4d &pos_3d = aprilgrid_corner_pos_3d[corner_id];

        pos_3d[0] = x_offset + x_corner_offsets[i];
        pos_3d[1] = y_offset + y_corner_offsets[i];
        pos_3d[2] = 0;
        pos_3d[3] = 1;
      }
    }
  }

  int num_vign_points = 5;
  int num_blocks = tagCols * tagRows * 2;

  aprilgrid_vignette_pos_3d.resize((num_blocks + tagCols + tagRows) *
                                   num_vign_points);

  for (int k = 0; k < num_vign_points; k++) {
    for (int i = 0; i < tagCols * tagRows; i++) {
      // const Eigen::Vector3d p0 = aprilgrid_corner_pos_3d[4 * i + 0];
      const Eigen::Vector4d p1 = aprilgrid_corner_pos_3d[4 * i + 1];
      const Eigen::Vector4d p2 = aprilgrid_corner_pos_3d[4 * i + 2];
      const Eigen::Vector4d p3 = aprilgrid_corner_pos_3d[4 * i + 3];

      double coeff = double(k + 1) / double(num_vign_points + 1);

      aprilgrid_vignette_pos_3d[k * num_blocks + 2 * i + 0] =
          (p1 + coeff * (p2 - p1));
      aprilgrid_vignette_pos_3d[k * num_blocks + 2 * i + 1] =
          (p2 + coeff * (p3 - p2));

      aprilgrid_vignette_pos_3d[k * num_blocks + 2 * i + 0][0] +=
          tagSize * tagSpacing / 2;
      aprilgrid_vignette_pos_3d[k * num_blocks + 2 * i + 1][1] +=
          tagSize * tagSpacing / 2;
    }
  }

  size_t curr_idx = num_blocks * num_vign_points;

  for (int k = 0; k < num_vign_points; k++) {
    for (int i = 0; i < tagCols; i++) {
      const Eigen::Vector4d p0 = aprilgrid_corner_pos_3d[4 * i + 0];
      const Eigen::Vector4d p1 = aprilgrid_corner_pos_3d[4 * i + 1];

      double coeff = double(k + 1) / double(num_vign_points + 1);

      aprilgrid_vignette_pos_3d[curr_idx + k * tagCols + i] =
          (p0 + coeff * (p1 - p0));

      aprilgrid_vignette_pos_3d[curr_idx + k * tagCols + i][1] -=
          tagSize * tagSpacing / 2;
    }
  }

  curr_idx += tagCols * num_vign_points;

  for (int k = 0; k < num_vign_points; k++) {
    for (int i = 0; i < tagRows; i++) {
      const Eigen::Vector4d p0 = aprilgrid_corner_pos_3d[4 * i * tagCols + 0];
      const Eigen::Vector4d p3 = aprilgrid_corner_pos_3d[4 * i * tagCols + 3];

      double coeff = double(k + 1) / double(num_vign_points + 1);

      aprilgrid_vignette_pos_3d[curr_idx + k * tagRows + i] =
          (p0 + coeff * (p3 - p0));

      aprilgrid_vignette_pos_3d[curr_idx + k * tagRows + i][0] -=
          tagSize * tagSpacing / 2;
    }
  }
}

}  // namespace basalt
