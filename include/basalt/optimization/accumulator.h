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

#ifndef BASALT_ACCUMULATOR_H
#define BASALT_ACCUMULATOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <chrono>
#include <unordered_map>

#include <basalt/utils/assert.h>

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

  inline VectorX solve() const { return H.ldlt().solve(b); }

  inline VectorX solveWithPrior(const MatrixX& H_prior, const VectorX& b_prior,
                                size_t pos) const {
    const int prior_size = b_prior.rows();

    BASALT_ASSERT_STREAM(
        H_prior.cols() == prior_size,
        "H_prior.cols() " << H_prior.cols() << " prior_size " << prior_size);
    BASALT_ASSERT_STREAM(
        H_prior.rows() == prior_size,
        "H_prior.rows() " << H_prior.rows() << " prior_size " << prior_size);

    MatrixX H_new = H;
    VectorX b_new = b;

    H_new.block(pos, pos, prior_size, prior_size) += H_prior;
    b_new.segment(pos, prior_size) += b_prior;

    return H_new.ldlt().solve(-b_new);
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
class SparseAccumulator {
 public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Triplet<Scalar> T;
  typedef Eigen::SparseMatrix<Scalar> SparseMatrix;

  template <int ROWS, int COLS, typename Derived>
  inline void addH(int si, int sj, const Eigen::MatrixBase<Derived>& data) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, ROWS, COLS);

    for (int i = 0; i < ROWS; i++) {
      for (int j = 0; j < COLS; j++) {
        triplets.emplace_back(si + i, sj + j, data(i, j));
      }
    }
  }

  template <int ROWS, typename Derived>
  inline void addB(int i, const Eigen::MatrixBase<Derived>& data) {
    b.template segment<ROWS>(i) += data;
  }

  inline VectorX solve() const {
    SparseMatrix sm(b.rows(), b.rows());

    auto triplets_copy = triplets;
    for (int i = 0; i < b.rows(); i++) {
      triplets_copy.emplace_back(i, i, 0.000001);
    }

    sm.setFromTriplets(triplets_copy.begin(), triplets_copy.end());

    // Eigen::IOFormat CleanFmt(2);
    // std::cerr << "sm\n" << sm.toDense().format(CleanFmt) << std::endl;

    Eigen::SimplicialLDLT<SparseMatrix> chol(sm);
    return chol.solve(-b);
    // return sm.toDense().ldlt().solve(-b);
  }

  inline void reset(int opt_size) {
    triplets.clear();
    b.setZero(opt_size);
  }

  inline void join(const SparseAccumulator<Scalar>& other) {
    triplets.reserve(triplets.size() + other.triplets.size());
    triplets.insert(triplets.end(), other.triplets.begin(),
                    other.triplets.end());
    b += other.b;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  std::vector<T> triplets;
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

  inline VectorX solve(Scalar alpha = 1e-6) const {
    SparseMatrix sm(b.rows(), b.rows());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<T> triplets;
    triplets.reserve(hash_map.size() * 36 + b.rows());

    if (alpha > 0)
      for (int i = 0; i < b.rows(); i++) {
        triplets.emplace_back(i, i, alpha);
      }

    for (const auto& kv : hash_map) {
      // if (kv.first[2] != kv.second.rows()) std::cerr << "rows mismatch" <<
      // std::endl;
      // if (kv.first[3] != kv.second.cols()) std::cerr << "cols mismatch" <<
      // std::endl;

      for (int i = 0; i < kv.second.rows(); i++) {
        for (int j = 0; j < kv.second.cols(); j++) {
          triplets.emplace_back(kv.first[0] + i, kv.first[1] + j,
                                kv.second(i, j));
        }
      }
    }

    sm.setFromTriplets(triplets.begin(), triplets.end());

    auto t2 = std::chrono::high_resolution_clock::now();

    // sm.diagonal() *= 1.01;

    // Eigen::IOFormat CleanFmt(2);
    // std::cerr << "sm\n" << sm.toDense().format(CleanFmt) << std::endl;

    VectorX res;

    if (iterative_solver) {
      Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                               Eigen::Lower | Eigen::Upper>
          cg;

      cg.setTolerance(tolerance);
      cg.compute(sm);
      res = cg.solve(b);
    } else {
      Eigen::SimplicialLDLT<SparseMatrix> chol(sm);
      res = chol.solve(b);
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    auto elapsed1 =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto elapsed2 =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);

    if (print_info) {
      std::cout << "Forming matrix: " << elapsed1.count() * 1e-6
                << "s. Solving linear system: " << elapsed2.count() * 1e-6
                << "s." << std::endl;
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
        hash_combine(seed, std::hash<int>()(c[i]));
      }
      return seed;
    }

    inline void hash_combine(std::size_t& seed, std::size_t value) const {
      seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
  };

  std::unordered_map<KeyT, MatrixX, KeyHash> hash_map;

  VectorX b;
};

}  // namespace basalt

#endif
