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

#include <Eigen/Dense>

#include <pangolin/gl/gldraw.h>

#include <basalt/utils/sophus_utils.hpp>

const uint8_t cam_color[3]{250, 0, 26};
const uint8_t state_color[3]{250, 0, 26};
const uint8_t pose_color[3]{0, 50, 255};
const uint8_t gt_color[3]{0, 171, 47};

inline void render_camera(const Eigen::Matrix4d& T_w_c, float lineWidth,
                          const uint8_t* color, float sizeFactor) {
  const float sz = sizeFactor;
  const float width = 640, height = 480, fx = 500, fy = 500, cx = 320, cy = 240;

  const Eigen::aligned_vector<Eigen::Vector3f> lines = {
      {0, 0, 0},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz}};

  glPushMatrix();
  glMultMatrixd(T_w_c.data());
  glColor3ubv(color);
  glLineWidth(lineWidth);
  pangolin::glDrawLines(lines);
  glPopMatrix();
}

inline void getcolor(float p, float np, float& r, float& g, float& b) {
  float inc = 4.0 / np;
  float x = p * inc;
  r = 0.0f;
  g = 0.0f;
  b = 0.0f;

  if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
    r = 1.0f;
  else if (4 <= x && x <= 5)
    r = x - 4;
  else if (1 <= x && x <= 2)
    r = 1.0f - (x - 1);

  if (1 <= x && x <= 3)
    g = 1.0f;
  else if (0 <= x && x <= 1)
    g = x - 0;
  else if (3 <= x && x <= 4)
    g = 1.0f - (x - 3);

  if (3 <= x && x <= 5)
    b = 1.0f;
  else if (2 <= x && x <= 3)
    b = x - 2;
  else if (5 <= x && x <= 6)
    b = 1.0f - (x - 5);
}
