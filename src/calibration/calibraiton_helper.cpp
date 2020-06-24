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

#include <basalt/calibration/calibration_helper.h>

#include <basalt/utils/apriltag.h>

#include <tbb/parallel_for.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>

#include <opengv/sac/Ransac.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#pragma GCC diagnostic pop

#include <opencv2/calib3d/calib3d.hpp>

namespace basalt {

template <class CamT>
bool estimateTransformation(
    const CamT &cam_calib,
    const Eigen::aligned_vector<Eigen::Vector2d> &corners,
    const std::vector<int> &corner_ids,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    Sophus::SE3d &T_target_camera, size_t &num_inliers) {
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (size_t i = 0; i < corners.size(); i++) {
    Eigen::Vector4d tmp;
    if (!cam_calib.unproject(corners[i], tmp)) {
      continue;
    }
    Eigen::Vector3d bearing = tmp.head<3>();
    Eigen::Vector3d point = aprilgrid_corner_pos_3d[corner_ids[i]].head<3>();
    bearing.normalize();

    bearingVectors.push_back(bearing);
    points.push_back(point);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 1.0 - cos(atan(sqrt(2.0) * 1 / cam_calib.getParam()[0]));
  ransac.max_iterations_ = 50;

  ransac.computeModel();

  T_target_camera =
      Sophus::SE3d(ransac.model_coefficients_.topLeftCorner<3, 3>(),
                   ransac.model_coefficients_.topRightCorner<3, 1>());

  num_inliers = ransac.inliers_.size();

  return ransac.inliers_.size() > 8;
}

void CalibHelper::detectCorners(const VioDatasetPtr &vio_data,
                                CalibCornerMap &calib_corners,
                                CalibCornerMap &calib_corners_rejected) {
  calib_corners.clear();
  calib_corners_rejected.clear();

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, vio_data->get_image_timestamps().size()),
      [&](const tbb::blocked_range<size_t> &r) {
        ApriltagDetector ad;

        for (size_t j = r.begin(); j != r.end(); ++j) {
          int64_t timestamp_ns = vio_data->get_image_timestamps()[j];
          const std::vector<ImageData> &img_vec =
              vio_data->get_image_data(timestamp_ns);

          for (size_t i = 0; i < img_vec.size(); i++) {
            if (img_vec[i].img.get()) {
              CalibCornerData ccd_good;
              CalibCornerData ccd_bad;
              ad.detectTags(*img_vec[i].img, ccd_good.corners,
                            ccd_good.corner_ids, ccd_good.radii,
                            ccd_bad.corners, ccd_bad.corner_ids, ccd_bad.radii);

              //                std::cout << "image (" << timestamp_ns << ","
              //                << i
              //                          << ")  detected " <<
              //                          ccd_good.corners.size()
              //                          << "corners (" <<
              //                          ccd_bad.corners.size()
              //                          << " rejected)" << std::endl;

              TimeCamId tcid(timestamp_ns, i);

              calib_corners.emplace(tcid, ccd_good);
              calib_corners_rejected.emplace(tcid, ccd_bad);
            }
          }
        }
      });
}

void CalibHelper::initCamPoses(
    const Calibration<double>::Ptr &calib,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    CalibCornerMap &calib_corners, CalibInitPoseMap &calib_init_poses) {
  calib_init_poses.clear();

  std::vector<TimeCamId> corners;
  corners.reserve(calib_corners.size());
  for (const auto &kv : calib_corners) {
    corners.emplace_back(kv.first);
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, corners.size()),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t j = r.begin(); j != r.end(); ++j) {
                        TimeCamId tcid = corners[j];
                        const CalibCornerData &ccd = calib_corners.at(tcid);

                        CalibInitPoseData cp;

                        computeInitialPose(calib, tcid.cam_id,
                                           aprilgrid_corner_pos_3d, ccd, cp);

                        calib_init_poses.emplace(tcid, cp);
                      }
                    });
}

bool CalibHelper::initializeIntrinsics(
    const Eigen::aligned_vector<Eigen::Vector2d> &corners,
    const std::vector<int> &corner_ids, const AprilGrid &aprilgrid, int cols,
    int rows, Eigen::Vector4d &init_intr) {
  // First, initialize the image center at the center of the image.

  Eigen::aligned_map<int, Eigen::Vector2d> id_to_corner;
  for (size_t i = 0; i < corner_ids.size(); i++) {
    id_to_corner[corner_ids[i]] = corners[i];
  }

  const double _xi = 1.0;
  const double _cu = cols / 2.0 - 0.5;
  const double _cv = rows / 2.0 - 0.5;

  /// Initialize some temporaries needed.
  double gamma0 = 0.0;
  double minReprojErr = std::numeric_limits<double>::max();

  // Now we try to find a non-radial line to initialize the focal length
  const size_t target_cols = aprilgrid.getTagCols();
  const size_t target_rows = aprilgrid.getTagRows();

  bool success = false;
  for (int tag_corner_offset = 0; tag_corner_offset < 2; tag_corner_offset++)
    for (size_t r = 0; r < target_rows; ++r) {
      // cv::Mat P(target.cols(); 4, CV_64F);

      Eigen::aligned_vector<Eigen::Vector4d> P;

      for (size_t c = 0; c < target_cols; ++c) {
        int tag_offset = (r * target_cols + c) << 2;

        for (int i = 0; i < 2; i++) {
          int corner_id = tag_offset + i + tag_corner_offset * 2;

          // std::cerr << corner_id << " ";

          if (id_to_corner.find(corner_id) != id_to_corner.end()) {
            const Eigen::Vector2d imagePoint = id_to_corner[corner_id];

            double u = imagePoint[0] - _cu;
            double v = imagePoint[1] - _cv;

            P.emplace_back(u, v, 0.5, -0.5 * (square(u) + square(v)));
          }
        }
      }

      // std::cerr << std::endl;

      const int MIN_CORNERS = 8;
      // MIN_CORNERS is an arbitrary threshold for the number of corners
      if (P.size() > MIN_CORNERS) {
        // Resize P to fit with the count of valid points.

        Eigen::Map<Eigen::Matrix4Xd> P_mat((double *)P.data(), 4, P.size());

        // std::cerr << "P_mat\n" << P_mat.transpose() << std::endl;

        Eigen::MatrixXd P_mat_t = P_mat.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            P_mat_t, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // std::cerr << "U\n" << svd.matrixU() << std::endl;
        // std::cerr << "V\n" << svd.matrixV() << std::endl;
        // std::cerr << "singularValues\n" << svd.singularValues() <<
        // std::endl;

        Eigen::Vector4d C = svd.matrixV().col(3);
        // std::cerr << "C\n" << C.transpose() << std::endl;
        // std::cerr << "P*res\n" << P_mat.transpose() * C << std::endl;

        double t = square(C(0)) + square(C(1)) + C(2) * C(3);
        if (t < 0) {
          continue;
        }

        // check that line image is not radial
        double d = sqrt(1.0 / t);
        double nx = C(0) * d;
        double ny = C(1) * d;
        if (hypot(nx, ny) > 0.95) {
          // std::cerr << "hypot(nx, ny) " << hypot(nx, ny) << std::endl;
          continue;
        }

        double nz = sqrt(1.0 - square(nx) - square(ny));
        double gamma = fabs(C(2) * d / nz);

        Eigen::Matrix<double, 5, 1> calib;
        calib << 0.5 * gamma, 0.5 * gamma, _cu, _cv, 0.5 * _xi;
        // std::cerr << "gamma " << gamma << std::endl;

        UnifiedCamera<double> cam_calib(calib);

        size_t num_inliers;
        Sophus::SE3d T_target_camera;
        if (!estimateTransformation(cam_calib, corners, corner_ids,
                                    aprilgrid.aprilgrid_corner_pos_3d,
                                    T_target_camera, num_inliers)) {
          continue;
        }

        double reprojErr = 0.0;
        size_t numReprojected = computeReprojectionError(
            cam_calib, corners, corner_ids, aprilgrid.aprilgrid_corner_pos_3d,
            T_target_camera, reprojErr);

        // std::cerr << "numReprojected " << numReprojected << " reprojErr "
        //          << reprojErr / numReprojected << std::endl;

        if (numReprojected > MIN_CORNERS) {
          double avgReprojErr = reprojErr / numReprojected;

          if (avgReprojErr < minReprojErr) {
            minReprojErr = avgReprojErr;
            gamma0 = gamma;
            success = true;
          }
        }

      }  // If this observation has enough valid corners
    }    // For each row in the image.

  if (success) init_intr << 0.5 * gamma0, 0.5 * gamma0, _cu, _cv;

  return success;
}

bool CalibHelper::initializeIntrinsicsPinhole(
    const std::vector<CalibCornerData *> pinhole_corners,
    const AprilGrid &aprilgrid, int cols, int rows,
    Eigen::Vector4d &init_intr) {
  // First, initialize the image center at the center of the image.

  const double _cu = cols / 2.0 - 0.5;
  const double _cv = rows / 2.0 - 0.5;

  // Z. Zhang, A Flexible New Technique for Camera Calibration, PAMI 2000

  size_t nImages = pinhole_corners.size();

  //  Eigen::MatrixXd A(2 * nImages, 2);
  //  Eigen::VectorXd b(2 * nImages);

  Eigen::MatrixXd A(nImages * 2, 2);
  Eigen::VectorXd b(nImages * 2, 1);

  int i = 0;

  for (const CalibCornerData *ccd : pinhole_corners) {
    const auto &corners = ccd->corners;
    const auto &corner_ids = ccd->corner_ids;

    std::vector<cv::Point2f> M(corners.size()), imagePoints(corners.size());
    for (size_t j = 0; j < corners.size(); ++j) {
      M.at(j) =
          cv::Point2f(aprilgrid.aprilgrid_corner_pos_3d[corner_ids[j]][0],
                      aprilgrid.aprilgrid_corner_pos_3d[corner_ids[j]][1]);

      //    std::cout << "corner "
      //              <<
      //              aprilgrid.aprilgrid_corner_pos_3d[corner_ids[j]].transpose()
      //              << std::endl;

      imagePoints.at(j) = cv::Point2f(corners[j][0], corners[j][1]);
    }

    cv::Mat H = cv::findHomography(M, imagePoints);

    if (H.empty()) return false;

    // std::cout << H << std::endl;

    H.at<double>(0, 0) -= H.at<double>(2, 0) * _cu;
    H.at<double>(0, 1) -= H.at<double>(2, 1) * _cu;
    H.at<double>(0, 2) -= H.at<double>(2, 2) * _cu;
    H.at<double>(1, 0) -= H.at<double>(2, 0) * _cv;
    H.at<double>(1, 1) -= H.at<double>(2, 1) * _cv;
    H.at<double>(1, 2) -= H.at<double>(2, 2) * _cv;

    double h[3], v[3], d1[3], d2[3];
    double n[4] = {0, 0, 0, 0};

    for (int j = 0; j < 3; ++j) {
      double t0 = H.at<double>(j, 0);
      double t1 = H.at<double>(j, 1);
      h[j] = t0;
      v[j] = t1;
      d1[j] = (t0 + t1) * 0.5;
      d2[j] = (t0 - t1) * 0.5;
      n[0] += t0 * t0;
      n[1] += t1 * t1;
      n[2] += d1[j] * d1[j];
      n[3] += d2[j] * d2[j];
    }

    for (int j = 0; j < 4; ++j) {
      n[j] = 1.0 / sqrt(n[j]);
    }

    for (int j = 0; j < 3; ++j) {
      h[j] *= n[0];
      v[j] *= n[1];
      d1[j] *= n[2];
      d2[j] *= n[3];
    }

    A(i * 2, 0) = h[0] * v[0];
    A(i * 2, 1) = h[1] * v[1];
    A(i * 2 + 1, 0) = d1[0] * d2[0];
    A(i * 2 + 1, 1) = d1[1] * d2[1];
    b(i * 2, 0) = -h[2] * v[2];
    b(i * 2 + 1, 0) = -d1[2] * d2[2];

    i++;
  }

  Eigen::Vector2d f = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  double fx = sqrt(fabs(1.0 / f(0)));
  double fy = sqrt(fabs(1.0 / f(1)));

  init_intr << fx, fy, _cu, _cv;

  return true;
}

void CalibHelper::computeInitialPose(
    const Calibration<double>::Ptr &calib, size_t cam_id,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    const CalibCornerData &cd, CalibInitPoseData &cp) {
  if (cd.corners.size() < 8) {
    cp.num_inliers = 0;
    return;
  }

  bool success;
  size_t num_inliers;

  std::visit(
      [&](const auto &cam) {
        Sophus::SE3d T_target_camera;
        success = estimateTransformation(cam, cd.corners, cd.corner_ids,
                                         aprilgrid_corner_pos_3d, cp.T_a_c,
                                         num_inliers);
      },
      calib->intrinsics[cam_id].variant);

  if (success) {
    Eigen::Matrix4d T_c_a_init = cp.T_a_c.inverse().matrix();

    std::vector<bool> proj_success;
    calib->intrinsics[cam_id].project(aprilgrid_corner_pos_3d, T_c_a_init,
                                      cp.reprojected_corners, proj_success);

    cp.num_inliers = num_inliers;
  } else {
    cp.num_inliers = 0;
  }
}

size_t CalibHelper::computeReprojectionError(
    const UnifiedCamera<double> &cam_calib,
    const Eigen::aligned_vector<Eigen::Vector2d> &corners,
    const std::vector<int> &corner_ids,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    const Sophus::SE3d &T_target_camera, double &error) {
  size_t num_projected = 0;
  error = 0;

  Eigen::Matrix4d T_camera_target = T_target_camera.inverse().matrix();

  for (size_t i = 0; i < corners.size(); i++) {
    Eigen::Vector4d p_cam =
        T_camera_target * aprilgrid_corner_pos_3d[corner_ids[i]];
    Eigen::Vector2d res;
    cam_calib.project(p_cam, res);
    res -= corners[i];

    num_projected++;
    error += res.norm();
  }

  return num_projected;
}
}  // namespace basalt
