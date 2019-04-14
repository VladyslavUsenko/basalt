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
#ifndef BASALT_TEST_UTILS_H
#define BASALT_TEST_UTILS_H

#include <Eigen/Dense>

template <typename Derived1, typename Derived2, typename F>
void test_jacobian_code(const std::string& name,
                        const Eigen::MatrixBase<Derived1>& Ja, F func,
                        const Eigen::MatrixBase<Derived2>& x0,
                        double eps = 1e-8, double max_norm = 1e-4) {
  typedef typename Derived1::Scalar Scalar;

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Jn = Ja;
  Jn.setZero();

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> inc = x0;
  for (int i = 0; i < Jn.cols(); i++) {
    inc.setZero();
    inc[i] += eps;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fpe = func(x0 + inc);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fme = func(x0 - inc);

    Jn.col(i) = (fpe - fme) / (2 * eps);
  }

  Scalar diff = (Ja - Jn).norm();

  if (diff > max_norm || !Ja.allFinite()) {
    std::cerr << name << std::endl;
    std::cerr << "Numeric Jacobian is different from analytic. Norm difference "
              << diff << std::endl;
    std::cerr << "Ja\n" << Ja << std::endl;
    std::cerr << "Jn\n" << Jn << std::endl;
  } else {
    // std::cout << name << std::endl;
    //    std::cout << "Success" << std::endl;
    //    std::cout << "Ja\n" << Ja << std::endl;
    //    std::cout << "Jn\n" << Jn << std::endl;
  }
}

#endif  // BASALT_TEST_UTILS_H
