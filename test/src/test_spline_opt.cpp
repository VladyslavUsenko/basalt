
#include <basalt/optimization/spline_optimize.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(SplineOpt, SplineOptTest) {
  int num_knots = 15;

  basalt::CalibAccelBias<double> accel_bias_full;
  accel_bias_full.setRandom();

  basalt::CalibGyroBias<double> gyro_bias_full;
  gyro_bias_full.setRandom();

  Eigen::Vector3d g(0, 0, -9.81);

  basalt::Se3Spline<5> gt_spline(int64_t(2e9));
  gt_spline.genRandomTrajectory(num_knots);

  basalt::SplineOptimization<5, double> spline_opt(int64_t(2e9));

  int64_t pose_dt_ns = 1e8;
  for (int64_t t_ns = pose_dt_ns / 2; t_ns < gt_spline.maxTimeNs();
       t_ns += pose_dt_ns) {
    Sophus::SE3d pose_gt = gt_spline.pose(t_ns);

    spline_opt.addPoseMeasurement(t_ns, pose_gt);
  }

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = 0; t_ns < gt_spline.maxTimeNs(); t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() * (gt_spline.transAccelWorld(t_ns) + g);

    // accel_body + accel_bias = (I + scale) * meas

    spline_opt.addAccelMeasurement(
        t_ns, accel_bias_full.invertCalibration(accel_body));
  }

  for (int64_t t_ns = 0; t_ns < gt_spline.maxTimeNs(); t_ns += dt_ns) {
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    spline_opt.addGyroMeasurement(
        t_ns, gyro_bias_full.invertCalibration(rot_vel_body));
  }

  spline_opt.resetCalib(0, {});

  spline_opt.initSpline(gt_spline);
  spline_opt.setG(g + Eigen::Vector3d::Random() / 10);
  spline_opt.init();

  double error;
  double reprojection_error;
  int num_inliers;
  for (int i = 0; i < 10; i++)
    spline_opt.optimize(false, true, false, false, true, false, 0.002, 1e-10,
                        error, num_inliers, reprojection_error);

  ASSERT_TRUE(
      spline_opt.getAccelBias().getParam().isApprox(accel_bias_full.getParam()))
      << "spline_opt.getCalib().accel_bias "
      << spline_opt.getGyroBias().getParam().transpose() << " and accel_bias "
      << accel_bias_full.getParam().transpose() << " are not the same";

  ASSERT_TRUE(
      spline_opt.getGyroBias().getParam().isApprox(gyro_bias_full.getParam()))
      << "spline_opt.getCalib().gyro_bias "
      << spline_opt.getGyroBias().getParam().transpose() << " and gyro_bias "
      << gyro_bias_full.getParam().transpose() << " are not the same";

  ASSERT_TRUE(spline_opt.getG().isApprox(g))
      << "spline_opt.getG() " << spline_opt.getG().transpose() << " and g "
      << g.transpose() << " are not the same";

  for (int64_t t_ns = 0; t_ns < gt_spline.maxTimeNs(); t_ns += 1e7) {
    Sophus::SE3d pose_gt = gt_spline.pose(t_ns);
    Sophus::SE3d pose = spline_opt.getSpline().pose(t_ns);

    Eigen::Vector3d pos_gt = pose_gt.translation();
    Eigen::Vector3d pos = pose.translation();

    Eigen::Quaterniond quat_gt = pose_gt.unit_quaternion();
    Eigen::Quaterniond quat = pose.unit_quaternion();

    Eigen::Vector3d accel_gt = gt_spline.transAccelWorld(t_ns);
    Eigen::Vector3d accel = spline_opt.getSpline().transAccelWorld(t_ns);

    Eigen::Vector3d gyro_gt = gt_spline.rotVelBody(t_ns);
    Eigen::Vector3d gyro = spline_opt.getSpline().rotVelBody(t_ns);

    ASSERT_TRUE(pos_gt.isApprox(pos)) << "pos_gt and pos are not the same";

    ASSERT_TRUE(quat_gt.angularDistance(quat) < 1e-2)
        << "quat_gt and quat are not the same";

    ASSERT_TRUE(accel_gt.isApprox(accel))
        << "accel_gt and accel are not the same";

    ASSERT_TRUE(gyro_gt.isApprox(gyro)) << "gyro_gt and gyro are not the same";
  }
}
