

#include <basalt/imu/preintegration.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/utils/ba_utils.h>
#include <basalt/vi_estimator/sc_ba_base.h>
#include <basalt/linearization/imu_block.hpp>

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

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(VioTestSuite, ImuNullspace2Test) {
  int num_knots = 15;

  Eigen::Vector3d bg, ba;
  bg = Eigen::Vector3d::Random() / 100;
  ba = Eigen::Vector3d::Random() / 10;

  basalt::IntegratedImuMeasurement<double> imu_meas(0, bg, ba);

  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  basalt::PoseVelBiasState<double> state0, state1, state1_gt;

  state0.t_ns = 0;
  state0.T_w_i = gt_spline.pose(int64_t(0));
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));
  state0.bias_gyro = bg;
  state0.bias_accel = ba;

  Eigen::Vector3d accel_cov, gyro_cov;
  accel_cov.setConstant(accel_std_dev * accel_std_dev);
  gyro_cov.setConstant(gyro_std_dev * gyro_std_dev);

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(1e8);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::g);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    basalt::ImuData<double> data;
    data.accel = accel_body + ba;
    data.gyro = rot_vel_body + bg;

    data.accel[0] += accel_noise_dist(gen);
    data.accel[1] += accel_noise_dist(gen);
    data.accel[2] += accel_noise_dist(gen);

    data.gyro[0] += gyro_noise_dist(gen);
    data.gyro[1] += gyro_noise_dist(gen);
    data.gyro[2] += gyro_noise_dist(gen);

    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas.integrate(data, accel_cov, gyro_cov);
  }

  state1.t_ns = imu_meas.get_dt_ns();
  state1.T_w_i = gt_spline.pose(imu_meas.get_dt_ns()) *
                 Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  state1.vel_w_i = gt_spline.transVelWorld(imu_meas.get_dt_ns()) +
                   Sophus::Vector3d::Random() / 10;
  state1.bias_gyro = bg;
  state1.bias_accel = ba;

  Eigen::Vector3d gyro_weight_sqrt;
  gyro_weight_sqrt.setConstant(1e3);

  Eigen::Vector3d accel_weight_sqrt;
  accel_weight_sqrt.setConstant(1e3);

  Eigen::aligned_map<int64_t, basalt::IntegratedImuMeasurement<double>>
      imu_meas_vec;
  Eigen::aligned_map<int64_t, basalt::PoseVelBiasStateWithLin<double>>
      frame_states;
  Eigen::aligned_map<int64_t, basalt::PoseStateWithLin<double>> frame_poses;

  imu_meas_vec[state0.t_ns] = imu_meas;
  frame_states[state0.t_ns] = basalt::PoseVelBiasStateWithLin<double>(state0);
  frame_states[state1.t_ns] = basalt::PoseVelBiasStateWithLin<double>(state1);

  int asize = 30;

  basalt::AbsOrderMap aom;
  aom.total_size = 30;
  aom.items = 2;
  aom.abs_order_map[state0.t_ns] = std::make_pair(0, 15);
  aom.abs_order_map[state1.t_ns] = std::make_pair(15, 15);

  basalt::ImuLinData<double> ild = {basalt::constants::g,
                                    gyro_weight_sqrt,
                                    accel_weight_sqrt,
                                    {std::make_pair(state0.t_ns, &imu_meas)}};

  basalt::DenseAccumulator<double> accum;
  accum.reset(aom.total_size);

  basalt::ImuBlock<double> ib(&imu_meas, &ild, aom);
  double e0 = ib.linearizeImu(frame_states);
  ib.add_dense_H_b(accum);

  // Check quadratic approximation
  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd rand_inc;
    rand_inc.setRandom(asize);
    rand_inc.normalize();
    rand_inc /= 10000;

    auto frame_states_copy = frame_states;
    frame_states_copy[state0.t_ns].applyInc(rand_inc.segment<15>(0));
    frame_states_copy[state1.t_ns].applyInc(rand_inc.segment<15>(15));

    double imu_error_u, bg_error_u, ba_error_u;
    basalt::ScBundleAdjustmentBase<double>::computeImuError(
        aom, imu_error_u, bg_error_u, ba_error_u, frame_states_copy,
        imu_meas_vec, gyro_weight_sqrt.array().square(),
        accel_weight_sqrt.array().square(), basalt::constants::g);

    double e1 = imu_error_u + bg_error_u + ba_error_u - e0;

    double e2 = 0.5 * rand_inc.transpose() * accum.getH() * rand_inc;
    e2 += rand_inc.transpose() * accum.getB();

    EXPECT_LE(std::abs(e1 - e2), 2e-2) << "e1 " << e1 << " e2 " << e2;
  }

  std::cout << "=========================================" << std::endl;
  Eigen::VectorXd null_res =
      basalt::ScBundleAdjustmentBase<double>::checkNullspace(
          accum.getH(), accum.getB(), aom, frame_states, frame_poses);
  std::cout << "=========================================" << std::endl;

  EXPECT_LE(std::abs(null_res[0]), 1e-8);
  EXPECT_LE(std::abs(null_res[1]), 1e-8);
  EXPECT_LE(std::abs(null_res[2]), 1e-8);
  EXPECT_LE(std::abs(null_res[5]), 1e-6);
}
#endif

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(VioTestSuite, ImuNullspace3Test) {
  int num_knots = 15;

  Eigen::Vector3d bg, ba;
  bg = Eigen::Vector3d::Random() / 100;
  ba = Eigen::Vector3d::Random() / 10;

  basalt::IntegratedImuMeasurement<double> imu_meas1(0, bg, ba);

  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  basalt::PoseVelBiasState<double> state0, state1, state2;

  state0.t_ns = 0;
  state0.T_w_i = gt_spline.pose(int64_t(0));
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));
  state0.bias_gyro = bg;
  state0.bias_accel = ba;

  Eigen::Vector3d accel_cov, gyro_cov;
  accel_cov.setConstant(accel_std_dev * accel_std_dev);
  gyro_cov.setConstant(gyro_std_dev * gyro_std_dev);

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(1e9);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::g);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    basalt::ImuData<double> data;
    data.accel = accel_body + ba;
    data.gyro = rot_vel_body + bg;

    data.accel[0] += accel_noise_dist(gen);
    data.accel[1] += accel_noise_dist(gen);
    data.accel[2] += accel_noise_dist(gen);

    data.gyro[0] += gyro_noise_dist(gen);
    data.gyro[1] += gyro_noise_dist(gen);
    data.gyro[2] += gyro_noise_dist(gen);

    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas1.integrate(data, accel_cov, gyro_cov);
  }

  basalt::IntegratedImuMeasurement<double> imu_meas2(imu_meas1.get_dt_ns(), bg,
                                                     ba);
  for (int64_t t_ns = imu_meas1.get_dt_ns() + dt_ns / 2;
       t_ns < int64_t(2e9);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::g);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    basalt::ImuData<double> data;
    data.accel = accel_body + ba;
    data.gyro = rot_vel_body + bg;

    data.accel[0] += accel_noise_dist(gen);
    data.accel[1] += accel_noise_dist(gen);
    data.accel[2] += accel_noise_dist(gen);

    data.gyro[0] += gyro_noise_dist(gen);
    data.gyro[1] += gyro_noise_dist(gen);
    data.gyro[2] += gyro_noise_dist(gen);

    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas2.integrate(data, accel_cov, gyro_cov);
  }

  state1.t_ns = imu_meas1.get_dt_ns();
  state1.T_w_i = gt_spline.pose(state1.t_ns) *
                 Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  state1.vel_w_i =
      gt_spline.transVelWorld(state1.t_ns) + Sophus::Vector3d::Random() / 10;
  state1.bias_gyro = bg;
  state1.bias_accel = ba;

  state2.t_ns = imu_meas1.get_dt_ns() + imu_meas2.get_dt_ns();
  state2.T_w_i = gt_spline.pose(state2.t_ns) *
                 Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  state2.vel_w_i =
      gt_spline.transVelWorld(state2.t_ns) + Sophus::Vector3d::Random() / 10;
  state2.bias_gyro = bg;
  state2.bias_accel = ba;

  Eigen::Vector3d gyro_weight_sqrt;
  gyro_weight_sqrt.setConstant(1e3);

  Eigen::Vector3d accel_weight_sqrt;
  accel_weight_sqrt.setConstant(1e3);

  Eigen::aligned_map<int64_t, basalt::IntegratedImuMeasurement<double>>
      imu_meas_vec;
  Eigen::aligned_map<int64_t, basalt::PoseVelBiasStateWithLin<double>>
      frame_states;
  Eigen::aligned_map<int64_t, basalt::PoseStateWithLin<double>> frame_poses;

  imu_meas_vec[imu_meas1.get_start_t_ns()] = imu_meas1;
  imu_meas_vec[imu_meas2.get_start_t_ns()] = imu_meas2;
  frame_states[state0.t_ns] = basalt::PoseVelBiasStateWithLin<double>(state0);
  frame_states[state1.t_ns] = basalt::PoseVelBiasStateWithLin<double>(state1);
  frame_states[state2.t_ns] = basalt::PoseVelBiasStateWithLin<double>(state2);

  int asize = 45;

  basalt::AbsOrderMap aom;
  aom.total_size = asize;
  aom.items = 2;
  aom.abs_order_map[state0.t_ns] = std::make_pair(0, 15);
  aom.abs_order_map[state1.t_ns] = std::make_pair(15, 15);
  aom.abs_order_map[state2.t_ns] = std::make_pair(30, 15);

  basalt::ImuLinData<double> ild = {basalt::constants::g,
                                    gyro_weight_sqrt,
                                    accel_weight_sqrt,
                                    {
                                        std::make_pair(state0.t_ns, &imu_meas1),
                                        std::make_pair(state1.t_ns, &imu_meas2),
                                    }};

  basalt::DenseAccumulator<double> accum;
  accum.reset(aom.total_size);

  basalt::ImuBlock<double> ib1(&imu_meas1, &ild, aom),
      ib2(&imu_meas2, &ild, aom);
  ib1.linearizeImu(frame_states);
  ib2.linearizeImu(frame_states);
  ib1.add_dense_H_b(accum);
  ib2.add_dense_H_b(accum);

  std::cout << "=========================================" << std::endl;
  Eigen::VectorXd null_res =
      basalt::ScBundleAdjustmentBase<double>::checkNullspace(
          accum.getH(), accum.getB(), aom, frame_states, frame_poses);
  std::cout << "=========================================" << std::endl;

  EXPECT_LE(std::abs(null_res[0]), 1e-8);
  EXPECT_LE(std::abs(null_res[1]), 1e-8);
  EXPECT_LE(std::abs(null_res[2]), 1e-8);
  EXPECT_LE(std::abs(null_res[5]), 1e-6);
}
#endif

TEST(VioTestSuite, RelPoseTest) {
  Sophus::SE3d T_w_i_h = Sophus::se3_expd(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_i_t = Sophus::se3_expd(Sophus::Vector6d::Random());

  Sophus::SE3d T_i_c_h = Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  Sophus::SE3d T_i_c_t = Sophus::se3_expd(Sophus::Vector6d::Random() / 10);

  Sophus::Matrix6d d_rel_d_h, d_rel_d_t;

  Sophus::SE3d T_t_h_sophus = basalt::computeRelPose(
      T_w_i_h, T_i_c_h, T_w_i_t, T_i_c_t, &d_rel_d_h, &d_rel_d_t);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_h", d_rel_d_h,
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_w_h_new = T_w_i_h;
          basalt::PoseState<double>::incPose(x, T_w_h_new);

          Sophus::SE3d T_t_h_sophus_new =
              basalt::computeRelPose(T_w_h_new, T_i_c_h, T_w_i_t, T_i_c_t);

          return Sophus::se3_logd(T_t_h_sophus_new * T_t_h_sophus.inverse());
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_t", d_rel_d_t,
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_w_t_new = T_w_i_t;
          basalt::PoseState<double>::incPose(x, T_w_t_new);

          Sophus::SE3d T_t_h_sophus_new =
              basalt::computeRelPose(T_w_i_h, T_i_c_h, T_w_t_new, T_i_c_t);
          return Sophus::se3_logd(T_t_h_sophus_new * T_t_h_sophus.inverse());
        },
        x0);
  }
}

TEST(VioTestSuite, LinearizePointsTest) {
  basalt::ExtendedUnifiedCamera<double> cam =
      basalt::ExtendedUnifiedCamera<double>::getTestProjections()[0];

  basalt::Keypoint<double> kpt_pos;

  Eigen::Vector4d point3d;
  cam.unproject(Eigen::Vector2d::Random() * 50, point3d);
  kpt_pos.direction = basalt::StereographicParam<double>::project(point3d);
  kpt_pos.inv_dist = 0.1231231;

  Sophus::SE3d T_w_h = Sophus::se3_expd(Sophus::Vector6d::Random() / 100);
  Sophus::SE3d T_w_t = Sophus::se3_expd(Sophus::Vector6d::Random() / 100);
  T_w_t.translation()[0] += 0.1;

  Sophus::SE3d T_t_h_sophus = T_w_t.inverse() * T_w_h;
  Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

  Eigen::Vector4d p_trans;
  p_trans = basalt::StereographicParam<double>::unproject(kpt_pos.direction);
  p_trans(3) = kpt_pos.inv_dist;

  p_trans = T_t_h * p_trans;

  basalt::KeypointObservation<double> kpt_obs;
  cam.project(p_trans, kpt_obs.pos);

  Eigen::Vector2d res;
  Eigen::Matrix<double, 2, 6> d_res_d_xi;
  Eigen::Matrix<double, 2, 3> d_res_d_p;

  basalt::linearizePoint(kpt_obs.pos, kpt_pos, T_t_h, cam, res, &d_res_d_xi,
                         &d_res_d_p);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_xi", d_res_d_xi,
        [&](const Sophus::Vector6d& x) {
          Eigen::Matrix4d T_t_h_new =
              (Sophus::se3_expd(x) * T_t_h_sophus).matrix();

          Eigen::Vector2d res;
          basalt::linearizePoint(kpt_obs.pos, kpt_pos, T_t_h_new, cam, res);

          return res;
        },
        x0);
  }

  {
    Eigen::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_p", d_res_d_p,
        [&](const Eigen::Vector3d& x) {
          basalt::Keypoint kpt_pos_new = kpt_pos;

          kpt_pos_new.direction += x.head<2>();
          kpt_pos_new.inv_dist += x[2];

          Eigen::Vector2d res;
          basalt::linearizePoint(kpt_obs.pos, kpt_pos_new, T_t_h, cam, res);

          return res;
        },
        x0);
  }
}
