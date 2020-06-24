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

#include <unordered_set>

#include <basalt/utils/keypoints.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#pragma GCC diagnostic pop

namespace basalt {

// const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

const static char pattern_31_x_a[256] = {
    8,   4,   -11, 7,   2,   1,   -2,  -13, -13, 10,  -13, -11, 7,   -4,  -13,
    -9,  12,  -3,  -6,  11,  4,   5,   3,   -8,  -2,  -13, -7,  -4,  -10, 5,
    5,   1,   9,   4,   2,   -4,  -8,  4,   0,   -13, -3,  -6,  8,   0,   7,
    -13, 10,  -6,  10,  -13, -13, 3,   5,   -1,  3,   2,   -13, -13, -13, -7,
    6,   -9,  -2,  -12, 3,   -7,  -3,  2,   -11, -1,  5,   -4,  -9,  -12, 10,
    7,   -7,  -4,  7,   -7,  -13, -3,  7,   -13, 1,   2,   -4,  -1,  7,   1,
    9,   -1,  -13, 7,   12,  6,   5,   2,   3,   2,   9,   -8,  -11, 1,   6,
    2,   6,   3,   7,   -11, -10, -5,  -10, 8,   4,   -10, 4,   -2,  -5,  7,
    -9,  -5,  8,   -9,  1,   7,   -2,  11,  -12, 3,   5,   0,   -9,  0,   -1,
    5,   3,   -13, -5,  -4,  6,   -7,  -13, 1,   4,   -2,  2,   -2,  4,   -6,
    -3,  7,   4,   -13, 7,   7,   -7,  -8,  -13, 2,   10,  -6,  8,   2,   -11,
    -12, -11, 5,   -2,  -1,  -13, -10, -3,  2,   -9,  -4,  -4,  -6,  6,   -13,
    11,  7,   -1,  -4,  -7,  -13, -7,  -8,  -5,  -13, 1,   1,   9,   5,   -1,
    -9,  -1,  -13, 8,   2,   7,   -10, -10, 4,   3,   -4,  5,   4,   -9,  0,
    -12, 3,   -10, 8,   -8,  2,   10,  6,   -7,  -3,  -1,  -3,  -8,  4,   2,
    6,   3,   11,  -3,  4,   2,   -10, -13, -13, 6,   0,   -13, -9,  -13, 5,
    2,   -1,  9,   11,  3,   -1,  3,   -13, 5,   8,   7,   -10, 7,   9,   7,
    -1};

const static char pattern_31_y_a[256] = {
    -3,  2,   9,   -12, -13, -7,  -10, -13, -3,  4,   -8,  7,   7,   -5,  2,
    0,   -6,  6,   -13, -13, 7,   -3,  -7,  -7,  11,  12,  3,   2,   -12, -12,
    -6,  0,   11,  7,   -1,  -12, -5,  11,  -8,  -2,  -2,  9,   12,  9,   -5,
    -6,  7,   -3,  -9,  8,   0,   3,   7,   7,   -10, -4,  0,   -7,  3,   12,
    -10, -1,  -5,  5,   -10, -7,  -2,  9,   -13, 6,   -3,  -13, -6,  -10, 2,
    12,  -13, 9,   -1,  6,   11,  7,   -8,  -7,  -3,  -6,  3,   -13, 1,   -1,
    1,   -9,  -13, 7,   -5,  3,   -13, -12, 8,   6,   -12, 4,   12,  12,  -9,
    3,   3,   -3,  8,   -5,  11,  -8,  5,   -1,  -6,  12,  -2,  0,   -8,  -6,
    -13, -13, -8,  -11, -8,  -4,  1,   -6,  -9,  7,   5,   -4,  12,  7,   2,
    11,  5,   -4,  9,   -7,  5,   6,   6,   -10, 1,   -2,  -12, -13, 1,   -10,
    -13, 5,   -2,  9,   1,   -8,  -4,  11,  6,   4,   -5,  -5,  -3,  -12, -2,
    -13, 0,   -3,  -13, -8,  -11, -2,  9,   -3,  -13, 6,   12,  -11, -3,  11,
    11,  -5,  12,  -8,  1,   -12, -2,  5,   -1,  7,   5,   0,   12,  -8,  11,
    -3,  -10, 1,   -11, -13, -13, -10, -8,  -6,  12,  2,   -13, -13, 9,   3,
    1,   2,   -10, -13, -12, 2,   6,   8,   10,  -9,  -13, -7,  -2,  2,   -5,
    -9,  -1,  -1,  0,   -11, -4,  -6,  7,   12,  0,   -1,  3,   8,   -6,  -9,
    7,   -6,  5,   -3,  0,   4,   -6,  0,   8,   9,   -4,  4,   3,   -7,  0,
    -6};

const static char pattern_31_x_b[256] = {
    9,   7,  -8, 12,  2,   1,  -2,  -11, -12, 11,  -8,  -9,  12,  -3,  -12, -7,
    12,  -2, -4, 12,  5,   10, 6,   -6,  -1,  -8,  -5,  -3,  -6,  6,   7,   4,
    11,  4,  4,  -2,  -7,  9,  1,   -8,  -2,  -4,  10,  1,   11,  -11, 12,  -6,
    12,  -8, -8, 7,   10,  1,  5,   3,   -13, -12, -11, -4,  12,  -7,  0,   -7,
    8,   -4, -1, 5,   -5,  0,  5,   -4,  -9,  -8,  12,  12,  -6,  -3,  12,  -5,
    -12, -2, 12, -11, 12,  3,  -2,  1,   8,   3,   12,  -1,  -10, 10,  12,  7,
    6,   2,  4,  12,  10,  -7, -4,  2,   7,   3,   11,  8,   9,   -6,  -5,  -3,
    -9,  12, 6,  -8,  6,   -2, -5,  10,  -8,  -5,  9,   -9,  1,   9,   -1,  12,
    -6,  7,  10, 2,   -5,  2,  1,   7,   6,   -8,  -3,  -3,  8,   -6,  -5,  3,
    8,   2,  12, 0,   9,   -3, -1,  12,  5,   -9,  8,   7,   -7,  -7,  -12, 3,
    12,  -6, 9,  2,   -10, -7, -10, 11,  -1,  0,   -12, -10, -2,  3,   -4,  -3,
    -2,  -4, 6,  -5,  12,  12, 0,   -3,  -6,  -8,  -6,  -6,  -4,  -8,  5,   10,
    10,  10, 1,  -6,  1,   -8, 10,  3,   12,  -5,  -8,  8,   8,   -3,  10,  5,
    -4,  3,  -6, 4,   -10, 12, -6,  3,   11,  8,   -6,  -3,  -1,  -3,  -8,  12,
    3,   11, 7,  12,  -3,  4,  2,   -8,  -11, -11, 11,  1,   -9,  -6,  -8,  8,
    3,   -1, 11, 12,  3,   0,  4,   -10, 12,  9,   8,   -10, 12,  10,  12,  0};

const static char pattern_31_y_b[256] = {
    5,   -12, 2,   -13, 12,  6,   -4,  -8,  -9,  9,   -9,  12,  6,   0,  -3,
    5,   -1,  12,  -8,  -8,  1,   -3,  12,  -2,  -10, 10,  -3,  7,   11, -7,
    -1,  -5,  -13, 12,  4,   7,   -10, 12,  -13, 2,   3,   -9,  7,   3,  -10,
    0,   1,   12,  -4,  -12, -4,  8,   -7,  -12, 6,   -10, 5,   12,  8,  7,
    8,   -6,  12,  5,   -13, 5,   -7,  -11, -13, -1,  2,   12,  6,   -4, -3,
    12,  5,   4,   2,   1,   5,   -6,  -7,  -12, 12,  0,   -13, 9,   -6, 12,
    6,   3,   5,   12,  9,   11,  10,  3,   -6,  -13, 3,   9,   -6,  -8, -4,
    -2,  0,   -8,  3,   -4,  10,  12,  0,   -6,  -11, 7,   7,   12,  2,  12,
    -8,  -2,  -13, 0,   -2,  1,   -4,  -11, 4,   12,  8,   8,   -13, 12, 7,
    -9,  -8,  9,   -3,  -12, 0,   12,  -2,  10,  -4,  -13, 12,  -6,  3,  -5,
    1,   -11, -7,  -5,  6,   6,   1,   -8,  -8,  9,   3,   7,   -8,  8,  3,
    -9,  -5,  8,   12,  9,   -5,  11,  -13, 2,   0,   -10, -7,  9,   11, 5,
    6,   -2,  7,   -2,  7,   -13, -8,  -9,  5,   10,  -13, -13, -1,  -9, -13,
    2,   12,  -10, -6,  -6,  -9,  -7,  -13, 5,   -13, -3,  -12, -1,  3,  -9,
    1,   -8,  9,   12,  -5,  7,   -8,  -12, 5,   9,   5,   4,   3,   12, 11,
    -13, 12,  4,   6,   12,  1,   1,   1,   -13, -13, 4,   -2,  -3,  -2, 10,
    -9,  -1,  -2,  -8,  5,   10,  5,   5,   11,  -6,  -12, 9,   4,   -2, -2,
    -11};

void detectKeypointsMapping(const basalt::Image<const uint16_t>& img_raw,
                            KeypointsData& kd, int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U);

  uint8_t* dst = image.ptr();
  const uint16_t* src = img_raw.ptr;

  for (size_t i = 0; i < img_raw.size(); i++) {
    dst[i] = (src[i] >> 8);
  }

  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(image, points, num_features, 0.01, 8);

  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  for (size_t i = 0; i < points.size(); i++) {
    if (img_raw.InBounds(points[i].x, points[i].y, EDGE_THRESHOLD)) {
      kd.corners.emplace_back(points[i].x, points[i].y);
    }
  }
}

void detectKeypoints(
    const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd,
    int PATCH_SIZE, int num_points_cell,
    const Eigen::aligned_vector<Eigen::Vector2d>& current_points) {
  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
  const size_t x_stop = x_start + img_raw.w - PATCH_SIZE;

  const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
  const size_t y_stop = y_start + img_raw.h - PATCH_SIZE;

  // std::cerr << "x_start " << x_start << " x_stop " << x_stop << std::endl;
  // std::cerr << "y_start " << y_start << " y_stop " << y_stop << std::endl;

  Eigen::MatrixXi cells;
  cells.setZero(img_raw.h / PATCH_SIZE + 1, img_raw.w / PATCH_SIZE + 1);

  for (const Eigen::Vector2d& p : current_points) {
    if (p[0] >= x_start && p[1] >= y_start) {
      int x = (p[0] - x_start) / PATCH_SIZE;
      int y = (p[1] - y_start) / PATCH_SIZE;

      cells(y, x) += 1;
    }
  }

  for (size_t x = x_start; x < x_stop; x += PATCH_SIZE) {
    for (size_t y = y_start; y < y_stop; y += PATCH_SIZE) {
      if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0)
        continue;

      const basalt::Image<const uint16_t> sub_img_raw =
          img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

      cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);

      for (int y = 0; y < PATCH_SIZE; y++) {
        uchar* sub_ptr = subImg.ptr(y);
        for (int x = 0; x < PATCH_SIZE; x++) {
          sub_ptr[x] = (sub_img_raw(x, y) >> 8);
        }
      }

      int points_added = 0;
      int threshold = 40;

      while (points_added < num_points_cell && threshold >= 5) {
        std::vector<cv::KeyPoint> points;
        cv::FAST(subImg, points, threshold);

        std::sort(points.begin(), points.end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool {
                    return a.response > b.response;
                  });

        //        std::cout << "Detected " << points.size() << " points.
        //        Threshold "
        //                  << threshold << std::endl;

        for (size_t i = 0; i < points.size() && points_added < num_points_cell;
             i++)
          if (img_raw.InBounds(x + points[i].pt.x, y + points[i].pt.y,
                               EDGE_THRESHOLD)) {
            kd.corners.emplace_back(x + points[i].pt.x, y + points[i].pt.y);
            points_added++;
          }

        threshold /= 2;
      }
    }
  }

  // std::cout << "Total points: " << kd.corners.size() << std::endl;

  //  cv::TermCriteria criteria =
  //      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
  //  cv::Size winSize = cv::Size(5, 5);
  //  cv::Size zeroZone = cv::Size(-1, -1);
  //  cv::cornerSubPix(image, points, winSize, zeroZone, criteria);

  //  for (size_t i = 0; i < points.size(); i++) {
  //    if (img_raw.InBounds(points[i].pt.x, points[i].pt.y, EDGE_THRESHOLD)) {
  //      kd.corners.emplace_back(points[i].pt.x, points[i].pt.y);
  //    }
  //  }
}

void computeAngles(const basalt::Image<const uint16_t>& img_raw,
                   KeypointsData& kd, bool rotate_features) {
  kd.corner_angles.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    const Eigen::Vector2d& p = kd.corners[i];

    const int cx = p[0];
    const int cy = p[1];

    double angle = 0;

    if (rotate_features) {
      double m01 = 0, m10 = 0;
      for (int x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; x++) {
        for (int y = -HALF_PATCH_SIZE; y <= HALF_PATCH_SIZE; y++) {
          if (x * x + y * y <= HALF_PATCH_SIZE * HALF_PATCH_SIZE) {
            double val = img_raw(cx + x, cy + y);
            m01 += y * val;
            m10 += x * val;
          }
        }
      }

      angle = atan2(m01, m10);
    }

    kd.corner_angles[i] = angle;
  }
}

void computeDescriptors(const basalt::Image<const uint16_t>& img_raw,
                        KeypointsData& kd) {
  kd.corner_descriptors.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    std::bitset<256> descriptor;

    const Eigen::Vector2d& p = kd.corners[i];
    double angle = kd.corner_angles[i];

    int cx = p[0];
    int cy = p[1];

    Eigen::Rotation2Dd rot(angle);
    Eigen::Matrix2d mat_rot = rot.matrix();

    for (int i = 0; i < 256; i++) {
      Eigen::Vector2d va(pattern_31_x_a[i], pattern_31_y_a[i]),
          vb(pattern_31_x_b[i], pattern_31_y_b[i]);

      Eigen::Vector2i vva = (mat_rot * va).array().round().cast<int>();
      Eigen::Vector2i vvb = (mat_rot * vb).array().round().cast<int>();

      descriptor[i] =
          img_raw(cx + vva[0], cy + vva[1]) < img_raw(cx + vvb[0], cy + vvb[1]);
    }

    kd.corner_descriptors[i] = descriptor;
  }
}

void matchFastHelper(const std::vector<std::bitset<256>>& corner_descriptors_1,
                     const std::vector<std::bitset<256>>& corner_descriptors_2,
                     std::unordered_map<int, int>& matches, int threshold,
                     double test_dist) {
  matches.clear();

  for (size_t i = 0; i < corner_descriptors_1.size(); i++) {
    int best_idx = -1, best_dist = 500;
    int best2_dist = 500;

    for (size_t j = 0; j < corner_descriptors_2.size(); j++) {
      int dist = (corner_descriptors_1[i] ^ corner_descriptors_2[j]).count();

      if (dist <= best_dist) {
        best2_dist = best_dist;

        best_dist = dist;
        best_idx = j;
      } else if (dist < best2_dist) {
        best2_dist = dist;
      }
    }

    if (best_dist < threshold && best_dist * test_dist <= best2_dist) {
      matches.emplace(i, best_idx);
    }
  }
}

void matchDescriptors(const std::vector<std::bitset<256>>& corner_descriptors_1,
                      const std::vector<std::bitset<256>>& corner_descriptors_2,
                      std::vector<std::pair<int, int>>& matches, int threshold,
                      double dist_2_best) {
  matches.clear();

  std::unordered_map<int, int> matches_1_2, matches_2_1;
  matchFastHelper(corner_descriptors_1, corner_descriptors_2, matches_1_2,
                  threshold, dist_2_best);
  matchFastHelper(corner_descriptors_2, corner_descriptors_1, matches_2_1,
                  threshold, dist_2_best);

  for (const auto& kv : matches_1_2) {
    if (matches_2_1[kv.second] == kv.first) {
      matches.emplace_back(kv.first, kv.second);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  opengv::bearingVectors_t bearingVectors1, bearingVectors2;

  for (size_t i = 0; i < md.matches.size(); i++) {
    bearingVectors1.push_back(kd1.corners_3d[md.matches[i].first].head<3>());
    bearingVectors2.push_back(kd2.corners_3d[md.matches[i].second].head<3>());
  }

  // create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
                                                        bearingVectors2);
  // create a RANSAC object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // create a CentralRelativePoseSacProblem
  // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::STEWENIUS));
  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.max_iterations_ = 100;
  ransac.computeModel();

  // do non-linear refinement and add more inliers
  const size_t num_inliers_ransac = ransac.inliers_.size();

  adapter.sett12(ransac.model_coefficients_.topRightCorner<3, 1>());
  adapter.setR12(ransac.model_coefficients_.topLeftCorner<3, 3>());

  const opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  // Sanity check if the number of inliers decreased, but only warn if it is
  // by 3 or more, since some small fluctuation is expected.
  if (ransac.inliers_.size() + 2 < num_inliers_ransac) {
    std::cout << "Warning: non-linear refinement reduced the relative pose "
                 "ransac inlier count from "
              << num_inliers_ransac << " to " << ransac.inliers_.size() << "."
              << std::endl;
  }

  // get the result (normalize translation)
  md.T_i_j = Sophus::SE3d(
      nonlinear_transformation.topLeftCorner<3, 3>(),
      nonlinear_transformation.topRightCorner<3, 1>().normalized());

  if ((long)ransac.inliers_.size() >= ransac_min_inliers) {
    for (size_t i = 0; i < ransac.inliers_.size(); i++)
      md.inliers.emplace_back(md.matches[ransac.inliers_[i]]);
  }
}

}  // namespace basalt
