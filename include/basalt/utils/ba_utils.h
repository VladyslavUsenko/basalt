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

#include <basalt/vi_estimator/landmark_database.h>

namespace basalt {

template <class Scalar>
Sophus::SE3<Scalar> computeRelPose(
    const Sophus::SE3<Scalar>& T_w_i_h, const Sophus::SE3<Scalar>& T_i_c_h,
    const Sophus::SE3<Scalar>& T_w_i_t, const Sophus::SE3<Scalar>& T_i_c_t,
    Sophus::Matrix6<Scalar>* d_rel_d_h = nullptr,
    Sophus::Matrix6<Scalar>* d_rel_d_t = nullptr) {
  Sophus::SE3<Scalar> tmp2 = (T_i_c_t).inverse();

  Sophus::SE3<Scalar> T_t_i_h_i;
  T_t_i_h_i.so3() = T_w_i_t.so3().inverse() * T_w_i_h.so3();
  T_t_i_h_i.translation() =
      T_w_i_t.so3().inverse() * (T_w_i_h.translation() - T_w_i_t.translation());

  Sophus::SE3<Scalar> tmp = tmp2 * T_t_i_h_i;
  Sophus::SE3<Scalar> res = tmp * T_i_c_h;

  if (d_rel_d_h) {
    Sophus::Matrix3<Scalar> R = T_w_i_h.so3().inverse().matrix();

    Sophus::Matrix6<Scalar> RR;
    RR.setZero();
    RR.template topLeftCorner<3, 3>() = R;
    RR.template bottomRightCorner<3, 3>() = R;

    *d_rel_d_h = tmp.Adj() * RR;
  }

  if (d_rel_d_t) {
    Sophus::Matrix3<Scalar> R = T_w_i_t.so3().inverse().matrix();

    Sophus::Matrix6<Scalar> RR;
    RR.setZero();
    RR.template topLeftCorner<3, 3>() = R;
    RR.template bottomRightCorner<3, 3>() = R;

    *d_rel_d_t = -tmp2.Adj() * RR;
  }

  return res;
}

template <class Scalar, class CamT>
inline bool linearizePoint(
    const Eigen::Matrix<Scalar, 2, 1>& kpt_obs, const Keypoint<Scalar>& kpt_pos,
    const Eigen::Matrix<Scalar, 4, 4>& T_t_h, const CamT& cam,
    Eigen::Matrix<Scalar, 2, 1>& res,
    Eigen::Matrix<Scalar, 2, POSE_SIZE>* d_res_d_xi = nullptr,
    Eigen::Matrix<Scalar, 2, 3>* d_res_d_p = nullptr,
    Eigen::Matrix<Scalar, 4, 1>* proj = nullptr) {
  static_assert(std::is_same_v<typename CamT::Scalar, Scalar>);

  // Todo implement without jacobians
  Eigen::Matrix<Scalar, 4, 2> Jup;
  Eigen::Matrix<Scalar, 4, 1> p_h_3d;
  p_h_3d = StereographicParam<Scalar>::unproject(kpt_pos.direction, &Jup);
  p_h_3d[3] = kpt_pos.inv_dist;

  const Eigen::Matrix<Scalar, 4, 1> p_t_3d = T_t_h * p_h_3d;

  Eigen::Matrix<Scalar, 2, 4> Jp;
  bool valid = cam.project(p_t_3d, res, &Jp);
  valid &= res.array().isFinite().all();

  if (!valid) {
    //      std::cerr << " Invalid projection! kpt_pos.dir "
    //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
    //                kpt_pos.id
    //                << " idx " << kpt_obs.kpt_id << std::endl;

    //      std::cerr << "T_t_h\n" << T_t_h << std::endl;
    //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;
    //      std::cerr << "p_t_3d\n" << p_t_3d.transpose() << std::endl;

    return false;
  }

  if (proj) {
    proj->template head<2>() = res;
    (*proj)[2] = p_t_3d[3] / p_t_3d.template head<3>().norm();
  }
  res -= kpt_obs;

  if (d_res_d_xi) {
    Eigen::Matrix<Scalar, 4, POSE_SIZE> d_point_d_xi;
    d_point_d_xi.template topLeftCorner<3, 3>() =
        Eigen::Matrix<Scalar, 3, 3>::Identity() * kpt_pos.inv_dist;
    d_point_d_xi.template topRightCorner<3, 3>() =
        -Sophus::SO3<Scalar>::hat(p_t_3d.template head<3>());
    d_point_d_xi.row(3).setZero();

    *d_res_d_xi = Jp * d_point_d_xi;
  }

  if (d_res_d_p) {
    Eigen::Matrix<Scalar, 4, 3> Jpp;
    Jpp.setZero();
    Jpp.template block<3, 2>(0, 0) = T_t_h.template topLeftCorner<3, 4>() * Jup;
    Jpp.col(2) = T_t_h.col(3);

    *d_res_d_p = Jp * Jpp;
  }

  return true;
}

}  // namespace basalt
