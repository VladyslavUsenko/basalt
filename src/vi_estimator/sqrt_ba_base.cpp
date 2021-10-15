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

#include <basalt/vi_estimator/sqrt_ba_base.h>

#include <tbb/parallel_for.h>

namespace basalt {

template <class Scalar_>
Eigen::VectorXd SqrtBundleAdjustmentBase<Scalar_>::checkNullspace(
    const MargLinData<Scalar_>& mld,
    const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
        frame_states,
    const Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>>& frame_poses,
    bool verbose) {
  using Vec3d = Eigen::Vector3d;
  using VecXd = Eigen::VectorXd;
  using Mat3d = Eigen::Matrix3d;
  using MatXd = Eigen::MatrixXd;
  using SO3d = Sophus::SO3d;

  BASALT_ASSERT_STREAM(size_t(mld.H.cols()) == mld.order.total_size,
                       mld.H.cols() << " " << mld.order.total_size);
  size_t marg_size = mld.order.total_size;

  // Idea: We construct increments that we know should lie in the null-space of
  // the prior from marginalized observations (except for the initial pose prior
  // we set at initialization) --> shift global translations (x,y,z separately),
  // or global rotations (r,p,y separately); for VIO only yaw rotation shift is
  // in nullspace. Compared to a random increment, we expect them to stay small
  // (at initial level of the initial pose prior). If they increase over time,
  // we accumulate spurious information on unobservable degrees of freedom.
  //
  // Poses are cam-to-world and we use left-increment in optimization, so
  // perturbations are in world frame. Hence translational increments are
  // independent. For rotational increments we also need to rotate translations
  // and velocities, both of which are expressed in world frame. For
  // translations, we move the center of rotation to the camera center centroid
  // for better numerics.

  VecXd inc_x, inc_y, inc_z, inc_roll, inc_pitch, inc_yaw;
  inc_x.setZero(marg_size);
  inc_y.setZero(marg_size);
  inc_z.setZero(marg_size);
  inc_roll.setZero(marg_size);
  inc_pitch.setZero(marg_size);
  inc_yaw.setZero(marg_size);

  int num_trans = 0;
  Vec3d mean_trans;
  mean_trans.setZero();

  // Compute mean translation
  for (const auto& kv : mld.order.abs_order_map) {
    Vec3d trans;
    if (kv.second.second == POSE_SIZE) {
      mean_trans += frame_poses.at(kv.first)
                        .getPoseLin()
                        .translation()
                        .template cast<double>();
      num_trans++;
    } else if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      mean_trans += frame_states.at(kv.first)
                        .getStateLin()
                        .T_w_i.translation()
                        .template cast<double>();
      num_trans++;
    } else {
      std::cerr << "Unknown size of the state: " << kv.second.second
                << std::endl;
      std::abort();
    }
  }
  mean_trans /= num_trans;

  double eps = 0.01;

  // Compute nullspace increments
  for (const auto& kv : mld.order.abs_order_map) {
    inc_x(kv.second.first + 0) = eps;
    inc_y(kv.second.first + 1) = eps;
    inc_z(kv.second.first + 2) = eps;
    inc_roll(kv.second.first + 3) = eps;
    inc_pitch(kv.second.first + 4) = eps;
    inc_yaw(kv.second.first + 5) = eps;

    Vec3d trans;
    if (kv.second.second == POSE_SIZE) {
      trans = frame_poses.at(kv.first)
                  .getPoseLin()
                  .translation()
                  .template cast<double>();
    } else if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      trans = frame_states.at(kv.first)
                  .getStateLin()
                  .T_w_i.translation()
                  .template cast<double>();
    } else {
      BASALT_ASSERT(false);
    }

    // combine rotation with global translation to make it rotation around
    // translation centroid, not around origin (better numerics). Note that
    // velocities are not affected by global translation.
    trans -= mean_trans;

    // Jacobian of translation w.r.t. the rotation increment (one column each
    // for the 3 different increments)
    Mat3d J = -SO3d::hat(trans);
    J *= eps;

    inc_roll.template segment<3>(kv.second.first) = J.col(0);
    inc_pitch.template segment<3>(kv.second.first) = J.col(1);
    inc_yaw.template segment<3>(kv.second.first) = J.col(2);

    if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      Vec3d vel = frame_states.at(kv.first)
                      .getStateLin()
                      .vel_w_i.template cast<double>();

      // Jacobian of velocity w.r.t. the rotation increment (one column each
      // for the 3 different increments)
      Mat3d J_vel = -SO3d::hat(vel);
      J_vel *= eps;

      inc_roll.template segment<3>(kv.second.first + POSE_SIZE) = J_vel.col(0);
      inc_pitch.template segment<3>(kv.second.first + POSE_SIZE) = J_vel.col(1);
      inc_yaw.template segment<3>(kv.second.first + POSE_SIZE) = J_vel.col(2);
    }
  }

  inc_x.normalize();
  inc_y.normalize();
  inc_z.normalize();
  inc_roll.normalize();
  inc_pitch.normalize();
  inc_yaw.normalize();

  //  std::cout << "inc_x   " << inc_x.transpose() << std::endl;
  //  std::cout << "inc_y   " << inc_y.transpose() << std::endl;
  //  std::cout << "inc_z   " << inc_z.transpose() << std::endl;
  //  std::cout << "inc_yaw " << inc_yaw.transpose() << std::endl;

  VecXd inc_random;
  inc_random.setRandom(marg_size);
  inc_random.normalize();

  MatXd H_d;
  VecXd b_d;

  if (mld.is_sqrt) {
    MatXd H_sqrt_d = mld.H.template cast<double>();
    VecXd b_sqrt_d = mld.b.template cast<double>();

    H_d = H_sqrt_d.transpose() * H_sqrt_d;
    b_d = H_sqrt_d.transpose() * b_sqrt_d;

  } else {
    H_d = mld.H.template cast<double>();
    b_d = mld.b.template cast<double>();
  }

  VecXd xHx(7);
  VecXd xb(7);

  xHx[0] = inc_x.transpose() * H_d * inc_x;
  xHx[1] = inc_y.transpose() * H_d * inc_y;
  xHx[2] = inc_z.transpose() * H_d * inc_z;
  xHx[3] = inc_roll.transpose() * H_d * inc_roll;
  xHx[4] = inc_pitch.transpose() * H_d * inc_pitch;
  xHx[5] = inc_yaw.transpose() * H_d * inc_yaw;
  xHx[6] = inc_random.transpose() * H_d * inc_random;

  // b == J^T r, so the increments should also lie in left-nullspace of b
  xb[0] = inc_x.transpose() * b_d;
  xb[1] = inc_y.transpose() * b_d;
  xb[2] = inc_z.transpose() * b_d;
  xb[3] = inc_roll.transpose() * b_d;
  xb[4] = inc_pitch.transpose() * b_d;
  xb[5] = inc_yaw.transpose() * b_d;
  xb[6] = inc_random.transpose() * b_d;

  if (verbose) {
    std::cout << "nullspace x_trans: " << xHx[0] << " + " << xb[0] << std::endl;
    std::cout << "nullspace y_trans: " << xHx[1] << " + " << xb[1] << std::endl;
    std::cout << "nullspace z_trans: " << xHx[2] << " + " << xb[2] << std::endl;
    std::cout << "nullspace roll   : " << xHx[3] << " + " << xb[3] << std::endl;
    std::cout << "nullspace pitch  : " << xHx[4] << " + " << xb[4] << std::endl;
    std::cout << "nullspace yaw    : " << xHx[5] << " + " << xb[5] << std::endl;
    std::cout << "nullspace random : " << xHx[6] << " + " << xb[6] << std::endl;
  }

  return xHx + xb;
}

template <class Scalar_>
Eigen::VectorXd SqrtBundleAdjustmentBase<Scalar_>::checkEigenvalues(
    const MargLinData<Scalar_>& mld, bool verbose) {
  // Check EV of J^T J explicitly instead of doing SVD on J to easily notice if
  // we have negative EVs (numerically). We do this computation in double
  // precision to avoid any inaccuracies that come from the squaring.

  // For EV, we use SelfAdjointEigenSolver to avoid getting (numerically)
  // complex eigenvalues.
  Eigen::MatrixXd H;

  if (mld.is_sqrt) {
    Eigen::MatrixXd sqrt_H_double = mld.H.template cast<double>();
    H = sqrt_H_double.transpose() * sqrt_H_double;
  } else {
    H = mld.H.template cast<double>();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H);
  if (eigensolver.info() != Eigen::Success) {
    BASALT_LOG_FATAL("eigen solver failed");
  }

  if (verbose) {
    std::cout << "EV:\n" << eigensolver.eigenvalues() << std::endl;
  }

  return eigensolver.eigenvalues();
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class SqrtBundleAdjustmentBase<double>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class SqrtBundleAdjustmentBase<float>;
#endif

}  // namespace basalt
