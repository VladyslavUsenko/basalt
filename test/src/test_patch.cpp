

#include <basalt/optical_flow/patch.h>
#include <sophus/se2.hpp>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

struct SmoothFunction {
  template <typename Scalar>
  Scalar interp(const Eigen::Matrix<Scalar, 2, 1>& p) const {
    return sin(p[0] / 100.0 + p[1] / 20.0);
  }

  template <typename Scalar>
  Eigen::Matrix<Scalar, 3, 1> interpGrad(
      const Eigen::Matrix<Scalar, 2, 1>& p) const {
    return Eigen::Matrix<Scalar, 3, 1>(sin(p[0] / 100.0 + p[1] / 20.0),
                                       cos(p[0] / 100.0 + p[1] / 20.0) / 100.0,
                                       cos(p[0] / 100.0 + p[1] / 20.0) / 20.0);
  }

  template <typename Derived>
  BASALT_HOST_DEVICE inline bool InBounds(
      const Eigen::MatrixBase<Derived>& p,
      const typename Derived::Scalar border) const {
    UNUSED(p);
    UNUSED(border);
    return true;
  }
};

TEST(Patch, ImageInterpolateGrad) {
  Eigen::Vector2i offset(231, 123);

  SmoothFunction img;

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGrad<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) {
        return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x));
      },
      Eigen::Vector2d::Zero(), 1);
}

TEST(Patch, PatchSe2Jac) {
  Eigen::Vector2i offset(231, 123);

  SmoothFunction img_view;

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  using PatternT = basalt::Pattern52<double>;
  using PatchT = basalt::OpticalFlowPatch<double, PatternT>;

  double mean, mean2;

  PatchT::VectorP data, data2;
  PatchT::MatrixP3 J_se2;

  basalt::OpticalFlowPatch<double, basalt::Pattern52<double>>::setDataJacSe2(
      img_view, pd, mean, data, J_se2);

  basalt::OpticalFlowPatch<double, basalt::Pattern52<double>>::setData(
      img_view, pd, mean2, data2);

  EXPECT_NEAR(mean, mean2, 1e-8);
  EXPECT_TRUE(data.isApprox(data2));

  test_jacobian(
      "d_res_d_se2", J_se2,
      [&](const Eigen::Vector3d& x) {
        Sophus::SE2d transform = Sophus::SE2d::exp(x);

        double mean3;
        PatchT::VectorP data3;

        basalt::OpticalFlowPatch<double, basalt::Pattern52<double>>::setData(
            img_view, pd, mean3, data3, &transform);

        return data3;
      },
      Eigen::Vector3d::Zero());
}
