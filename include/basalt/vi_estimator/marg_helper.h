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

#include <Eigen/Dense>
#include <set>

namespace basalt {

template <class Scalar_>
class MargHelper {
 public:
  using Scalar = Scalar_;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  // Modifies abs_H and abs_b as a side effect.
  static void marginalizeHelperSqToSq(MatX& abs_H, VecX& abs_b,
                                      const std::set<int>& idx_to_keep,
                                      const std::set<int>& idx_to_marg,
                                      MatX& marg_H, VecX& marg_b);

  // Modifies abs_H and abs_b as a side effect.
  static void marginalizeHelperSqToSqrt(MatX& abs_H, VecX& abs_b,
                                        const std::set<int>& idx_to_keep,
                                        const std::set<int>& idx_to_marg,
                                        MatX& marg_sqrt_H, VecX& marg_sqrt_b);

  // Modifies Q2Jp and Q2r as a side effect.
  static void marginalizeHelperSqrtToSqrt(MatX& Q2Jp, VecX& Q2r,
                                          const std::set<int>& idx_to_keep,
                                          const std::set<int>& idx_to_marg,
                                          MatX& marg_sqrt_H, VecX& marg_sqrt_b);
};
}  // namespace basalt
