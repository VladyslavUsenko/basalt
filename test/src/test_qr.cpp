
#include <Eigen/Dense>
#include <iostream>

#include <basalt/vi_estimator/marg_helper.h>

#include "gtest/gtest.h"

TEST(QRTestSuite, QRvsLLT) {
  Eigen::MatrixXd J;
  J.setRandom(10, 6);

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(J);
  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
  Eigen::MatrixXd LT = (J.transpose() * J).llt().matrixU();

  std::cout << "J\n" << J << "\ncol_norms: " << J.colwise().norm() << std::endl;
  std::cout << "R\n" << R << "\ncol_norms: " << R.colwise().norm() << std::endl;
  std::cout << "LT\n"
            << LT << "\ncol_norms: " << LT.colwise().norm() << std::endl;
}

TEST(QRTestSuite, QRvsLLTRankDef) {
  Eigen::MatrixXd J;
  J.setRandom(10, 6);

  J.col(2) = J.col(4);

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(J);
  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
  Eigen::MatrixXd LT = (J.transpose() * J).llt().matrixU();

  std::cout << "J\n" << J << "\ncol_norms: " << J.colwise().norm() << std::endl;
  std::cout << "R\n" << R << "\ncol_norms: " << R.colwise().norm() << std::endl;
  std::cout << "LT\n"
            << LT << "\ncol_norms: " << LT.colwise().norm() << std::endl;
}

#ifdef BASALT_INSTANTIATIONS_DOUBLE
TEST(QRTestSuite, RankDefLeastSquares) {
  Eigen::MatrixXd J;
  Eigen::VectorXd r;
  J.setRandom(10, 6);
  r.setRandom(10);

  J.col(1) = J.col(4);

  auto decomp = J.completeOrthogonalDecomposition();

  Eigen::VectorXd original_solution = decomp.solve(r);

  std::cout << "full solution: " << original_solution.transpose() << std::endl;
  std::cout << "Rank " << decomp.rank() << std::endl;

  std::cout << "sol OR:\t" << original_solution.tail<4>().transpose()
            << std::endl;

  std::set<int> idx_to_marg = {0, 1};
  std::set<int> idx_to_keep = {2, 3, 4, 5};

  Eigen::VectorXd sol_sc, sol_sqrt_sc, sol_sqrt_sc2, sol_qr;

  // sqrt SC version
  {
    Eigen::MatrixXd H = J.transpose() * J;
    Eigen::VectorXd b = J.transpose() * r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;

    basalt::MargHelper<double>::marginalizeHelperSqToSqrt(
        H, b, idx_to_keep, idx_to_marg, marg_sqrt_H, marg_sqrt_b);

    auto dec = marg_sqrt_H.completeOrthogonalDecomposition();

    sol_sqrt_sc = dec.solve(marg_sqrt_b);
    std::cout << "sol SQ:\t" << sol_sqrt_sc.transpose() << std::endl;
    std::cout << "rank " << dec.rank() << std::endl;

    auto dec2 = (marg_sqrt_H.transpose() * marg_sqrt_H)
                    .completeOrthogonalDecomposition();

    sol_sqrt_sc2 = dec2.solve(marg_sqrt_H.transpose() * marg_sqrt_b);
    std::cout << "sol SH:\t" << sol_sqrt_sc2.transpose() << std::endl;
    std::cout << "rank " << dec2.rank() << std::endl;
  }

  // SC version
  {
    Eigen::MatrixXd H = J.transpose() * J;
    Eigen::VectorXd b = J.transpose() * r;

    Eigen::MatrixXd marg_H;
    Eigen::VectorXd marg_b;

    basalt::MargHelper<double>::marginalizeHelperSqToSq(
        H, b, idx_to_keep, idx_to_marg, marg_H, marg_b);

    auto dec = marg_H.completeOrthogonalDecomposition();

    sol_sc = dec.solve(marg_b);
    std::cout << "sol SC:\t" << sol_sc.transpose() << std::endl;
    std::cout << "rank " << dec.rank() << std::endl;
  }

  // QR version
  {
    Eigen::MatrixXd J1 = J;
    Eigen::VectorXd r1 = r;

    Eigen::MatrixXd marg_sqrt_H;
    Eigen::VectorXd marg_sqrt_b;

    basalt::MargHelper<double>::marginalizeHelperSqrtToSqrt(
        J1, r1, idx_to_keep, idx_to_marg, marg_sqrt_H, marg_sqrt_b);

    auto dec = marg_sqrt_H.completeOrthogonalDecomposition();

    sol_qr = dec.solve(marg_sqrt_b);
    std::cout << "sol QR:\t" << sol_qr.transpose() << std::endl;
    std::cout << "rank " << dec.rank() << std::endl;
  }

  EXPECT_TRUE(sol_qr.isApprox(sol_sc));
  EXPECT_TRUE(sol_qr.isApprox(sol_sqrt_sc2));
}
#endif
