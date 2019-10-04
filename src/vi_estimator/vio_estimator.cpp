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

#include <basalt/vi_estimator/vio_estimator.h>

#include <basalt/vi_estimator/keypoint_vio.h>
#include <basalt/vi_estimator/keypoint_vo.h>

namespace basalt {

VioEstimatorBase::Ptr VioEstimatorFactory::getVioEstimator(
    const VioConfig& config, const Calibration<double>& cam,
    const Eigen::Vector3d& g, bool use_imu) {
  VioEstimatorBase::Ptr res;

  if (use_imu) {
    res.reset(new KeypointVioEstimator(g, cam, config));
  } else {
    res.reset(new KeypointVoEstimator(cam, config));
  }

  return res;
}

double alignSVD(const std::vector<int64_t>& filter_t_ns,
                const Eigen::aligned_vector<Eigen::Vector3d>& filter_t_w_i,
                const std::vector<int64_t>& gt_t_ns,
                Eigen::aligned_vector<Eigen::Vector3d>& gt_t_w_i) {
  Eigen::aligned_vector<Eigen::Vector3d> est_associations;
  Eigen::aligned_vector<Eigen::Vector3d> gt_associations;

  for (size_t i = 0; i < filter_t_w_i.size(); i++) {
    int64_t t_ns = filter_t_ns[i];

    size_t j;
    for (j = 0; j < gt_t_ns.size(); j++) {
      if (gt_t_ns.at(j) > t_ns) break;
    }
    j--;

    if (j >= gt_t_ns.size() - 1) {
      continue;
    }

    double dt_ns = t_ns - gt_t_ns.at(j);
    double int_t_ns = gt_t_ns.at(j + 1) - gt_t_ns.at(j);

    BASALT_ASSERT_STREAM(dt_ns >= 0, "dt_ns " << dt_ns);
    BASALT_ASSERT_STREAM(int_t_ns > 0, "int_t_ns " << int_t_ns);

    // Skip if the interval between gt larger than 100ms
    if (int_t_ns > 1.1e8) continue;

    double ratio = dt_ns / int_t_ns;

    BASALT_ASSERT(ratio >= 0);
    BASALT_ASSERT(ratio < 1);

    Eigen::Vector3d gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1];

    gt_associations.emplace_back(gt);
    est_associations.emplace_back(filter_t_w_i[i]);
  }

  int num_kfs = est_associations.size();

  Eigen::Matrix<double, 3, Eigen::Dynamic> gt, est;
  gt.setZero(3, num_kfs);
  est.setZero(3, num_kfs);

  for (size_t i = 0; i < est_associations.size(); i++) {
    gt.col(i) = gt_associations[i];
    est.col(i) = est_associations[i];
  }

  Eigen::Vector3d mean_gt = gt.rowwise().mean();
  Eigen::Vector3d mean_est = est.rowwise().mean();

  gt.colwise() -= mean_gt;
  est.colwise() -= mean_est;

  Eigen::Matrix3d cov = gt * est.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
    S(2, 2) = -1;

  Eigen::Matrix3d rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3d trans = mean_gt - rot_gt_est * mean_est;

  Sophus::SE3d T_gt_est(rot_gt_est, trans);
  Sophus::SE3d T_est_gt = T_gt_est.inverse();

  for (size_t i = 0; i < gt_t_w_i.size(); i++) {
    gt_t_w_i[i] = T_est_gt * gt_t_w_i[i];
  }

  double error = 0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    est_associations[i] = T_gt_est * est_associations[i];
    Eigen::Vector3d res = est_associations[i] - gt_associations[i];

    error += res.transpose() * res;
  }

  error /= est_associations.size();
  error = std::sqrt(error);

  std::cout << "T_align\n" << T_gt_est.matrix() << std::endl;
  std::cout << "error " << error << std::endl;
  std::cout << "number of associations " << num_kfs << std::endl;

  return error;
}
}  // namespace basalt
