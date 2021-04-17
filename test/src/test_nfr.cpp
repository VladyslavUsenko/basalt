

#include <basalt/spline/se3_spline.h>
#include <basalt/utils/nfr.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

static const double accel_std_dev = 0.23;
static const double gyro_std_dev = 0.0027;

// Smaller noise for testing
// static const double accel_std_dev = 0.00023;
// static const double gyro_std_dev = 0.0000027;

std::random_device rd{};
std::mt19937 gen{rd()};

std::normal_distribution<> gyro_noise_dist{0, gyro_std_dev};
std::normal_distribution<> accel_noise_dist{0, accel_std_dev};

TEST(PreIntegrationTestSuite, RelPoseTest) {
  Sophus::SE3d T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_j = Sophus::se3_expd(Sophus::Vector6d::Random());

  Sophus::SE3d T_i_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) *
                       T_w_i.inverse() * T_w_j;

  Sophus::Matrix6d d_res_d_T_w_i, d_res_d_T_w_j;
  basalt::relPoseError(T_i_j, T_w_i, T_w_j, &d_res_d_T_w_i, &d_res_d_T_w_j);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_T_w_i", d_res_d_T_w_i,
        [&](const Sophus::Vector6d& x) {
          auto T_w_i_new = T_w_i;
          basalt::PoseState<double>::incPose(x, T_w_i_new);

          return basalt::relPoseError(T_i_j, T_w_i_new, T_w_j);
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_T_w_j", d_res_d_T_w_j,
        [&](const Sophus::Vector6d& x) {
          auto T_w_j_new = T_w_j;
          basalt::PoseState<double>::incPose(x, T_w_j_new);

          return basalt::relPoseError(T_i_j, T_w_i, T_w_j_new);
        },
        x0);
  }
}

TEST(PreIntegrationTestSuite, AbsPositionTest) {
  Sophus::SE3d T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random());

  Eigen::Vector3d pos = T_w_i.translation() + Eigen::Vector3d::Random() / 10;

  Sophus::Matrix<double, 3, 6> d_res_d_T_w_i;
  basalt::absPositionError(T_w_i, pos, &d_res_d_T_w_i);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_T_w_i", d_res_d_T_w_i,
        [&](const Sophus::Vector6d& x) {
          auto T_w_i_new = T_w_i;
          basalt::PoseState<double>::incPose(x, T_w_i_new);

          return basalt::absPositionError(T_w_i_new, pos);
        },
        x0);
  }
}

TEST(PreIntegrationTestSuite, YawTest) {
  Sophus::SE3d T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random());

  Eigen::Vector3d yaw_dir_body =
      T_w_i.so3().inverse() * Eigen::Vector3d::UnitX();

  T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) * T_w_i;

  Sophus::Matrix<double, 1, 6> d_res_d_T_w_i;
  basalt::yawError(T_w_i, yaw_dir_body, &d_res_d_T_w_i);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_T_w_i", d_res_d_T_w_i,
        [&](const Sophus::Vector6d& x) {
          auto T_w_i_new = T_w_i;
          basalt::PoseState<double>::incPose(x, T_w_i_new);

          double res = basalt::yawError(T_w_i_new, yaw_dir_body);

          return Eigen::Matrix<double, 1, 1>(res);
        },
        x0);
  }
}

TEST(PreIntegrationTestSuite, RollPitchTest) {
  Sophus::SE3d T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random());

  Sophus::SO3d R_w_i = T_w_i.so3();

  T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) * T_w_i;

  Sophus::Matrix<double, 2, 6> d_res_d_T_w_i;
  basalt::rollPitchError(T_w_i, R_w_i, &d_res_d_T_w_i);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_T_w_i", d_res_d_T_w_i,
        [&](const Sophus::Vector6d& x) {
          auto T_w_i_new = T_w_i;
          basalt::PoseState<double>::incPose(x, T_w_i_new);

          return basalt::rollPitchError(T_w_i_new, R_w_i);
        },
        x0);
  }
}
