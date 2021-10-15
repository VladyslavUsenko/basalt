

#include <basalt/linearization/linearization_base.hpp>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

template <class Scalar>
void get_vo_estimator(int num_frames,
                      basalt::BundleAdjustmentBase<Scalar>& estimator,
                      basalt::AbsOrderMap& aom) {
  static constexpr int POSE_SIZE = 6;

  estimator.huber_thresh = 0.5;
  estimator.obs_std_dev = 2.0;

  estimator.calib.T_i_c.emplace_back(
      Sophus::se3_expd(Sophus::Vector6d::Random() / 100));
  estimator.calib.T_i_c.emplace_back(
      Sophus::se3_expd(Sophus::Vector6d::Random() / 100));

  basalt::GenericCamera<Scalar> cam;
  cam.variant = basalt::KannalaBrandtCamera4<Scalar>::getTestProjections()[0];

  estimator.calib.intrinsics.emplace_back(cam);
  estimator.calib.intrinsics.emplace_back(cam);

  Eigen::MatrixXd points_3d;
  points_3d.setRandom(3, num_frames * 10);
  points_3d.row(2).array() += 5.0;

  aom.total_size = 0;

  for (int i = 0; i < num_frames; i++) {
    Sophus::SE3d T_w_i;
    T_w_i.so3() = Sophus::SO3d::exp(Eigen::Vector3d::Random() / 100);
    T_w_i.translation()[0] = i * 0.1;

    aom.abs_order_map[i] = std::make_pair(i * POSE_SIZE, POSE_SIZE);
    aom.total_size += POSE_SIZE;

    estimator.frame_poses[i] = basalt::PoseStateWithLin(i, T_w_i, false);

    for (int j = 0; j < 10; j++) {
      const int kp_idx = 10 * i + j;
      Eigen::Vector3d p3d = points_3d.col(kp_idx);
      basalt::Keypoint<Scalar> kpt;

      Sophus::SE3d T_c_w = estimator.calib.T_i_c[0].inverse() * T_w_i.inverse();
      Eigen::Vector3d p3d_cam = T_c_w * p3d;

      kpt.direction =
          basalt::StereographicParam<Scalar>::project(p3d_cam.homogeneous());
      kpt.inv_dist = 1.0 / p3d_cam.norm();
      kpt.host_kf_id = basalt::TimeCamId(i, 0);

      estimator.lmdb.addLandmark(kp_idx, kpt);

      for (const auto& [frame_id, frame_pose] : estimator.frame_poses) {
        for (int c = 0; c < 2; c++) {
          basalt::TimeCamId tcid(frame_id, c);
          Sophus::SE3d T_c_w = estimator.calib.T_i_c[c].inverse() *
                               frame_pose.getPose().inverse();

          Eigen::Vector3d p3d_cam = T_c_w * p3d;
          Eigen::Vector2d p2d_cam;
          cam.project(p3d_cam, p2d_cam);

          p2d_cam += Eigen::Vector2d::Random() / 100;

          basalt::KeypointObservation<Scalar> ko;
          ko.kpt_id = kp_idx;
          ko.pos = p2d_cam;

          estimator.lmdb.addObservation(tcid, ko);
        }
      }
    }
  }
}

template <class Scalar>
void get_vo_estimator_with_marg(int num_frames,
                                basalt::BundleAdjustmentBase<Scalar>& estimator,
                                basalt::AbsOrderMap& aom,
                                basalt::MargLinData<Scalar>& mld) {
  static constexpr int POSE_SIZE = 6;

  get_vo_estimator(num_frames, estimator, aom);

  mld.H.setIdentity(2 * POSE_SIZE, 2 * POSE_SIZE);
  mld.H *= 1e6;

  mld.b.setRandom(2 * POSE_SIZE);
  mld.b *= 10;

  mld.order.abs_order_map[0] = std::make_pair(0, POSE_SIZE);
  mld.order.abs_order_map[1] = std::make_pair(POSE_SIZE, POSE_SIZE);
  mld.order.total_size = 2 * POSE_SIZE;

  estimator.frame_poses[0].setLinTrue();
  estimator.frame_poses[1].setLinTrue();

  estimator.frame_poses[0].applyInc(Sophus::Vector6d::Random() / 100);
  estimator.frame_poses[1].applyInc(Sophus::Vector6d::Random() / 100);
}

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(LinearizationTestSuite, VoNoMargLinearizationTest) {
  using Scalar = double;
  static constexpr int POSE_SIZE = 6;
  static constexpr int NUM_FRAMES = 6;

  basalt::BundleAdjustmentBase<Scalar> estimator;
  basalt::AbsOrderMap aom;

  get_vo_estimator<Scalar>(NUM_FRAMES, estimator, aom);

  typename basalt::LinearizationBase<Scalar, POSE_SIZE>::Options options;
  options.lb_options.huber_parameter = estimator.huber_thresh;
  options.lb_options.obs_std_dev = estimator.obs_std_dev;

  Eigen::MatrixXd H_abs_qr, H_abs_sc, H_rel_sc;
  Eigen::VectorXd b_abs_qr, b_abs_sc, b_rel_sc;

  options.linearization_type = basalt::LinearizationType::ABS_QR;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_qr;

  l_abs_qr = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(&estimator,
                                                                  aom, options);
  Scalar error_abs_qr = l_abs_qr->linearizeProblem();
  l_abs_qr->performQR();

  l_abs_qr->get_dense_H_b(H_abs_qr, b_abs_qr);

  options.linearization_type = basalt::LinearizationType::ABS_SC;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_sc;
  l_abs_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(&estimator,
                                                                  aom, options);

  Scalar error_abs_sc = l_abs_sc->linearizeProblem();
  l_abs_sc->performQR();

  l_abs_sc->get_dense_H_b(H_abs_sc, b_abs_sc);

  options.linearization_type = basalt::LinearizationType::REL_SC;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_rel_sc;
  l_rel_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(&estimator,
                                                                  aom, options);

  Scalar error_rel_sc = l_rel_sc->linearizeProblem();
  l_rel_sc->performQR();

  l_rel_sc->get_dense_H_b(H_rel_sc, b_rel_sc);

  Scalar error_diff = std::abs(error_abs_qr - error_abs_sc);
  Scalar H_diff = (H_abs_qr - H_abs_sc).norm();
  Scalar b_diff = (b_abs_qr - b_abs_sc).norm();

  Scalar error_diff2 = std::abs(error_abs_qr - error_rel_sc);
  Scalar H_diff2 = (H_abs_qr - H_rel_sc).norm();
  Scalar b_diff2 = (b_abs_qr - b_rel_sc).norm();

  EXPECT_LE(error_diff, 1e-8);
  EXPECT_LE(H_diff, 1e-8);
  EXPECT_LE(b_diff, 1e-8);

  EXPECT_LE(error_diff2, 1e-8);
  EXPECT_LE(H_diff2, 1e-8);
  EXPECT_LE(b_diff2, 1e-8);
}
#endif

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(LinearizationTestSuite, VoMargLinearizationTest) {
  using Scalar = double;
  static constexpr int POSE_SIZE = 6;
  static constexpr int NUM_FRAMES = 6;

  basalt::BundleAdjustmentBase<Scalar> estimator;
  basalt::MargLinData<Scalar> mld;
  basalt::AbsOrderMap aom;

  get_vo_estimator_with_marg<Scalar>(NUM_FRAMES, estimator, aom, mld);

  typename basalt::LinearizationBase<Scalar, POSE_SIZE>::Options options;
  options.lb_options.huber_parameter = estimator.huber_thresh;
  options.lb_options.obs_std_dev = estimator.obs_std_dev;

  Eigen::MatrixXd H_abs_qr, H_abs_sc, H_rel_sc;
  Eigen::VectorXd b_abs_qr, b_abs_sc, b_rel_sc;

  options.linearization_type = basalt::LinearizationType::ABS_QR;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_qr;

  l_abs_qr = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
      &estimator, aom, options, &mld);
  Scalar error_abs_qr = l_abs_qr->linearizeProblem();
  l_abs_qr->performQR();

  l_abs_qr->get_dense_H_b(H_abs_qr, b_abs_qr);

  options.linearization_type = basalt::LinearizationType::ABS_SC;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_sc;
  l_abs_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
      &estimator, aom, options, &mld);

  Scalar error_abs_sc = l_abs_sc->linearizeProblem();
  l_abs_sc->performQR();

  l_abs_sc->get_dense_H_b(H_abs_sc, b_abs_sc);

  options.linearization_type = basalt::LinearizationType::REL_SC;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_rel_sc;
  l_rel_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
      &estimator, aom, options, &mld);

  Scalar error_rel_sc = l_rel_sc->linearizeProblem();
  l_rel_sc->performQR();

  l_rel_sc->get_dense_H_b(H_rel_sc, b_rel_sc);

  Scalar error_diff = std::abs(error_abs_qr - error_abs_sc);
  Scalar H_diff = (H_abs_qr - H_abs_sc).norm();
  Scalar b_diff = (b_abs_qr - b_abs_sc).norm();

  Scalar error_diff2 = std::abs(error_abs_qr - error_rel_sc);
  Scalar H_diff2 = (H_abs_qr - H_rel_sc).norm();
  Scalar b_diff2 = (b_abs_qr - b_rel_sc).norm();

  EXPECT_LE(error_diff, 1e-8);
  EXPECT_LE(H_diff, 1e-8);
  EXPECT_LE(b_diff, 1e-8);

  EXPECT_LE(error_diff2, 1e-8);
  EXPECT_LE(H_diff2, 1e-8);
  EXPECT_LE(b_diff2, 1e-8);
}
#endif

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(LinearizationTestSuite, VoMargBacksubstituteTest) {
  using Scalar = double;
  static constexpr int POSE_SIZE = 6;
  static constexpr int NUM_FRAMES = 6;

  basalt::BundleAdjustmentBase<Scalar> estimator;
  basalt::MargLinData<Scalar> mld;
  basalt::AbsOrderMap aom;

  get_vo_estimator_with_marg<Scalar>(NUM_FRAMES, estimator, aom, mld);

  basalt::BundleAdjustmentBase<Scalar> estimator2 = estimator,
                                       estimator3 = estimator;

  typename basalt::LinearizationBase<Scalar, POSE_SIZE>::Options options;
  options.lb_options.huber_parameter = estimator.huber_thresh;
  options.lb_options.obs_std_dev = estimator.obs_std_dev;

  Eigen::MatrixXd H_abs_qr, H_abs_sc, H_rel_sc;
  Eigen::VectorXd b_abs_qr, b_abs_sc, b_rel_sc;

  options.linearization_type = basalt::LinearizationType::ABS_QR;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_qr;

  l_abs_qr = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
      &estimator, aom, options, &mld);
  Scalar error_abs_qr = l_abs_qr->linearizeProblem();
  l_abs_qr->performQR();
  l_abs_qr->get_dense_H_b(H_abs_qr, b_abs_qr);

  options.linearization_type = basalt::LinearizationType::ABS_SC;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_sc;
  l_abs_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
      &estimator2, aom, options, &mld);

  Scalar error_abs_sc = l_abs_sc->linearizeProblem();
  l_abs_sc->performQR();
  l_abs_sc->get_dense_H_b(H_abs_sc, b_abs_sc);

  options.linearization_type = basalt::LinearizationType::REL_SC;
  std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_rel_sc;
  l_rel_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
      &estimator3, aom, options, &mld);

  Scalar error_rel_sc = l_rel_sc->linearizeProblem();
  l_rel_sc->performQR();
  l_rel_sc->get_dense_H_b(H_rel_sc, b_rel_sc);

  Scalar error_diff = std::abs(error_abs_qr - error_abs_sc);
  Scalar H_diff = (H_abs_qr - H_abs_sc).norm();
  Scalar b_diff = (b_abs_qr - b_abs_sc).norm();

  Scalar error_diff2 = std::abs(error_abs_qr - error_rel_sc);
  Scalar H_diff2 = (H_abs_qr - H_rel_sc).norm();
  Scalar b_diff2 = (b_abs_qr - b_rel_sc).norm();

  EXPECT_LE(error_diff, 1e-8);
  EXPECT_LE(H_diff, 1e-8);
  EXPECT_LE(b_diff, 1e-8);

  EXPECT_LE(error_diff2, 1e-8);
  EXPECT_LE(H_diff2, 1e-8);
  EXPECT_LE(b_diff2, 1e-8);

  const Eigen::VectorXd inc = -H_rel_sc.ldlt().solve(b_rel_sc);

  Scalar l_diff1 = l_abs_qr->backSubstitute(inc);
  Scalar l_diff2 = l_abs_sc->backSubstitute(inc);
  Scalar l_diff3 = l_rel_sc->backSubstitute(inc);

  EXPECT_LE(abs(l_diff1 - l_diff2), 1e-8);
  EXPECT_LE(abs(l_diff2 - l_diff3), 1e-8);

  Scalar error1, error2, error3;

  estimator.computeError(error1);
  estimator2.computeError(error2);
  estimator3.computeError(error3);

  EXPECT_LE(abs(error1 - error2), 1e-8);
  EXPECT_LE(abs(error1 - error3), 1e-8);
}
#endif

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(LinearizationTestSuite, VoMargSqrtLinearizationTest) {
  using Scalar = double;
  static constexpr int POSE_SIZE = 6;
  static constexpr int NUM_FRAMES = 6;

  basalt::BundleAdjustmentBase<Scalar> estimator;
  basalt::MargLinData<Scalar> mld;
  basalt::AbsOrderMap aom;

  get_vo_estimator_with_marg<Scalar>(NUM_FRAMES, estimator, aom, mld);

  typename basalt::LinearizationBase<Scalar, POSE_SIZE>::Options options;
  options.lb_options.huber_parameter = estimator.huber_thresh;
  options.lb_options.obs_std_dev = estimator.obs_std_dev;

  Eigen::MatrixXd H_abs_qr, H_abs_sc, H_rel_sc, Q2TJp_abs_qr, Q2TJp_rel_sc,
      Q2TJp_abs_sc;
  Eigen::VectorXd b_abs_qr, b_abs_sc, b_rel_sc, Q2Tr_abs_qr, Q2Tr_rel_sc,
      Q2Tr_abs_sc;

  Scalar error_abs_qr, error_abs_sc, error_rel_sc;

  {
    options.linearization_type = basalt::LinearizationType::ABS_QR;
    std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_qr;

    l_abs_qr = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
        &estimator, aom, options, &mld);
    error_abs_qr = l_abs_qr->linearizeProblem();
    l_abs_qr->performQR();
    l_abs_qr->get_dense_H_b(H_abs_qr, b_abs_qr);
    l_abs_qr->get_dense_Q2Jp_Q2r(Q2TJp_abs_qr, Q2Tr_abs_qr);
  }
  {
    options.linearization_type = basalt::LinearizationType::ABS_SC;
    std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_abs_sc;
    l_abs_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
        &estimator, aom, options, &mld);

    error_abs_sc = l_abs_sc->linearizeProblem();
    l_abs_sc->performQR();
    l_abs_sc->get_dense_H_b(H_abs_sc, b_abs_sc);
    l_abs_sc->get_dense_Q2Jp_Q2r(Q2TJp_abs_sc, Q2Tr_abs_sc);
  }

  {
    options.linearization_type = basalt::LinearizationType::REL_SC;
    std::unique_ptr<basalt::LinearizationBase<Scalar, POSE_SIZE>> l_rel_sc;
    l_rel_sc = basalt::LinearizationBase<Scalar, POSE_SIZE>::create(
        &estimator, aom, options, &mld);

    error_rel_sc = l_rel_sc->linearizeProblem();
    l_rel_sc->performQR();
    l_rel_sc->get_dense_H_b(H_rel_sc, b_rel_sc);
    l_rel_sc->get_dense_Q2Jp_Q2r(Q2TJp_rel_sc, Q2Tr_rel_sc);
  }

  Scalar error_diff = std::abs(error_abs_qr - error_abs_sc);
  Scalar H_diff = (H_abs_qr - H_abs_sc).norm();
  Scalar b_diff = (b_abs_qr - b_abs_sc).norm();

  Scalar error_diff2 = std::abs(error_abs_qr - error_rel_sc);
  Scalar H_diff2 = (H_abs_qr - H_rel_sc).norm();
  Scalar b_diff2 = (b_abs_qr - b_rel_sc).norm();

  EXPECT_LE(error_diff, 1e-8);
  EXPECT_LE(H_diff, 1e-8);
  EXPECT_LE(b_diff, 1e-8);

  EXPECT_LE(error_diff2, 1e-8);
  EXPECT_LE(H_diff2, 1e-8);
  EXPECT_LE(b_diff2, 1e-8);

  // Should hold for full rank problems
  Scalar H_diff3 = (Q2TJp_abs_qr.transpose() * Q2TJp_abs_qr - H_abs_qr).norm();
  Scalar b_diff3 = (Q2TJp_abs_qr.transpose() * Q2Tr_abs_qr - b_abs_qr).norm();
  EXPECT_LE(H_diff3, 1e-3);
  EXPECT_LE(b_diff3, 1e-5);

  Scalar H_diff4 = (Q2TJp_abs_sc.transpose() * Q2TJp_abs_sc - H_abs_sc).norm();
  Scalar b_diff4 = (Q2TJp_abs_sc.transpose() * Q2Tr_abs_sc - b_abs_sc).norm();
  EXPECT_LE(H_diff4, 1e-3);
  EXPECT_LE(b_diff4, 1e-5);

  Scalar H_diff5 = (Q2TJp_rel_sc.transpose() * Q2TJp_rel_sc - H_rel_sc).norm();
  Scalar b_diff5 = (Q2TJp_rel_sc.transpose() * Q2Tr_rel_sc - b_rel_sc).norm();
  EXPECT_LE(H_diff5, 1e-3);
  EXPECT_LE(b_diff5, 1e-5);
}
#endif
