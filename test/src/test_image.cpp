
#include <Eigen/Dense>

#include <basalt/utils/image.h>

#include "gtest/gtest.h"
#include "test_utils.h"

void setImageData(uint16_t *imageArray, int size) {
  double norm = RAND_MAX;
  norm /= (double)std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < size; i++) {
    imageArray[i] = (unsigned char)(rand() / norm);
  }
}

TEST(Pattern, ImageInterp) {
  Eigen::Vector2d offset(231.234242, 123.23424);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector3d vg = img.interpGrad(offset);

  Eigen::Matrix<double, 1, 2> J = vg.tail<2>();

  // std::cerr << "vg\n" << vg << std::endl;

  test_jacobian(
      "d_val_d_p", J,
      [&](const Eigen::Vector2d &x) {
        Eigen::Matrix<double, 1, 1> res;
        Eigen::Vector2d p1 = offset + x;
        res[0] = img.interpGrad(p1)[0];
        return res;
      },
      Eigen::Vector2d::Zero(), 1.0);
}

TEST(Image, ImageInterpolate) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  double eps = 1e-12;
  double threshold = 1e-8;

  {
    Eigen::Vector2i pi = offset;
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset;
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(1, 0);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(-eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(0, 1);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, -eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(1, 1);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(-eps, -eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }
}

TEST(Image, ImageInterpolateGrad) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d valGrad = img.interpGrad<double>(pd);
  Eigen::Matrix<double, 1, 2> J = valGrad.tail<2>();

  test_jacobian(
      "d_res_d_x", J,
      [&](const Eigen::Vector2d &x) {
        return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x));
      },
      Eigen::Vector2d::Zero(), 1);
}
