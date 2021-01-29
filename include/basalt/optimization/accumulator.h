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
#include <Eigen/Sparse>

#include <array>
#include <chrono>
#include <unordered_map>

#include <basalt/utils/assert.h>
#include <basalt/utils/hash.h>

#if defined(BASALT_USE_CHOLMOD)

#include <Eigen/CholmodSupport>

template <class T>
using SparseLLT = Eigen::CholmodSupernodalLLT<T>;

#else

template <class T>
using SparseLLT = Eigen::SimplicialLDLT<T>;

#endif

namespace basalt {

template <typename Scalar = double>
class DenseAccumulator {
 public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  template <int ROWS, int COLS, typename Derived>
  inline void addH(int i, int j, const Eigen::MatrixBase<Derived>& data) {
    BASALT_ASSERT_STREAM(i >= 0, "i " << i);
    BASALT_ASSERT_STREAM(j >= 0, "j " << j);

    BASALT_ASSERT_STREAM(i + ROWS <= H.cols(), "i " << i << " ROWS " << ROWS
                                                    << " H.rows() "
                                                    << H.rows());
    BASALT_ASSERT_STREAM(j + COLS <= H.rows(), "j " << j << " COLS " << COLS
                                                    << " H.cols() "
                                                    << H.cols());

    H.template block<ROWS, COLS>(i, j) += data;
  }

  template <int ROWS, typename Derived>
  inline void addB(int i, const Eigen::MatrixBase<Derived>& data) {
    BASALT_ASSERT_STREAM(i >= 0, "i " << i);

    BASALT_ASSERT_STREAM(i + ROWS <= H.cols(), "i " << i << " ROWS " << ROWS
                                                    << " H.rows() "
                                                    << H.rows());

    b.template segment<ROWS>(i) += data;
  }

  // inline VectorX solve() const { return H.ldlt().solve(b); }
  inline VectorX solve(const VectorX* diagonal) const {
    if (diagonal == nullptr) {
      return H.ldlt().solve(b);
    } else {
      MatrixX HH = H;
      HH.diagonal() += *diagonal;
      return HH.ldlt().solve(b);
    }
  }

  inline void reset(int opt_size) {
    H.setZero(opt_size, opt_size);
    b.setZero(opt_size);
  }

  inline void join(const DenseAccumulator<Scalar>& other) {
    H += other.H;
    b += other.b;
  }

  inline void print() {
    Eigen::IOFormat CleanFmt(2);
    std::cerr << "H\n" << H.format(CleanFmt) << std::endl;
    std::cerr << "b\n" << b.transpose().format(CleanFmt) << std::endl;
  }

  inline void setup_solver(){};
  inline VectorX Hdiagonal() const { return H.diagonal(); }

  inline const MatrixX& getH() const { return H; }
  inline const VectorX& getB() const { return b; }

  inline MatrixX& getH() { return H; }
  inline VectorX& getB() { return b; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  MatrixX H;
  VectorX b;
};

template <typename Scalar = double>
class SparseHashAccumulator {
 public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
  typedef Eigen::Triplet<Scalar> T;
  typedef Eigen::SparseMatrix<Scalar> SparseMatrix;

  template <int ROWS, int COLS, typename Derived>
  inline void addH(int si, int sj, const Eigen::MatrixBase<Derived>& data) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, ROWS, COLS);

    KeyT id;
    id[0] = si;
    id[1] = sj;
    id[2] = ROWS;
    id[3] = COLS;

    auto it = hash_map.find(id);

    if (it == hash_map.end()) {
      hash_map.emplace(id, data);
    } else {
      it->second += data;
    }
  }

  template <int ROWS, typename Derived>
  inline void addB(int i, const Eigen::MatrixBase<Derived>& data) {
    b.template segment<ROWS>(i) += data;
  }

  inline void setup_solver() {
    std::vector<T> triplets;
    triplets.reserve(hash_map.size() * 36 + b.rows());

    for (const auto& kv : hash_map) {
      for (int i = 0; i < kv.second.rows(); i++) {
        for (int j = 0; j < kv.second.cols(); j++) {
          triplets.emplace_back(kv.first[0] + i, kv.first[1] + j,
                                kv.second(i, j));
        }
      }
    }

    for (int i = 0; i < b.rows(); i++) {
      triplets.emplace_back(i, i, std::numeric_limits<double>::min());
    }

    smm = SparseMatrix(b.rows(), b.rows());
    smm.setFromTriplets(triplets.begin(), triplets.end());
  }

  inline VectorX Hdiagonal() const { return smm.diagonal(); }

  inline VectorX& getB() { return b; }

  inline VectorX solve(const VectorX* diagonal) const {
    auto t2 = std::chrono::high_resolution_clock::now();

    SparseMatrix sm = smm;
    if (diagonal) sm.diagonal() += *diagonal;

    VectorX res;

    if (iterative_solver) {
      // NOTE: since we have to disable Eigen's parallelization with OpenMP
      // (interferes with TBB), the current CG is single-threaded, and we
      // can expect a substantial speedup by switching to a parallel
      // implementation of CG.
      Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                               Eigen::Lower | Eigen::Upper>
          cg;

      cg.setTolerance(tolerance);
      cg.compute(sm);
      res = cg.solve(b);
    } else {
      SparseLLT<SparseMatrix> chol(sm);
      res = chol.solve(b);
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    auto elapsed2 =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);

    if (print_info) {
      std::cout << "Solving linear system: " << elapsed2.count() * 1e-6 << "s."
                << std::endl;
    }

    return res;
  }

  inline void reset(int opt_size) {
    hash_map.clear();
    b.setZero(opt_size);
  }

  inline void join(const SparseHashAccumulator<Scalar>& other) {
    for (const auto& kv : other.hash_map) {
      auto it = hash_map.find(kv.first);

      if (it == hash_map.end()) {
        hash_map.emplace(kv.first, kv.second);
      } else {
        it->second += kv.second;
      }
    }

    b += other.b;
  }

  double tolerance = 1e-4;
  bool iterative_solver = false;
  bool print_info = false;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  using KeyT = std::array<int, 4>;

  struct KeyHash {
    inline size_t operator()(const KeyT& c) const {
      size_t seed = 0;
      for (int i = 0; i < 4; i++) {
        hash_combine(seed, c[i]);
      }
      return seed;
    }
  };

  std::unordered_map<KeyT, MatrixX, KeyHash> hash_map;

  VectorX b;

  SparseMatrix smm;
};

}  // namespace basalt
