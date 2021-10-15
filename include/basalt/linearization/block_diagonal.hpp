#pragma once

#include <unordered_map>

#include <Eigen/Dense>

#include <basalt/utils/cast_utils.hpp>

namespace basalt {

// TODO: expand IndexedBlocks to small class / struct that also holds info on
// block size and number of blocks, so we don't have to pass it around all the
// time and we can directly implement things link adding diagonal and matrix
// vector products in this sparse block diagonal matrix.

// map of camera index to block; used to represent sparse block diagonal matrix
template <typename Scalar>
using IndexedBlocks =
    std::unordered_map<size_t,
                       Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>;

// scale dimensions of JTJ as you would do for jacobian scaling of J beforehand,
// with diagonal scaling matrix D: For jacobain we would use JD, so for JTJ we
// use DJTJD.
template <typename Scalar>
void scale_jacobians(
    IndexedBlocks<Scalar>& block_diagonal, size_t num_blocks, size_t block_size,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scaling_vector) {
  BASALT_ASSERT(num_blocks * block_size ==
                unsigned_cast(scaling_vector.size()));
  for (auto& [cam_idx, block] : block_diagonal) {
    auto D =
        scaling_vector.segment(block_size * cam_idx, block_size).asDiagonal();
    block = D * block * D;
  }
}

// add diagonal to block-diagonal matrix; missing blocks are assumed to be 0
template <typename Scalar>
void add_diagonal(IndexedBlocks<Scalar>& block_diagonal, size_t num_blocks,
                  size_t block_size,
                  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& diagonal) {
  BASALT_ASSERT(num_blocks * block_size == unsigned_cast(diagonal.size()));
  for (size_t idx = 0; idx < num_blocks; ++idx) {
    auto [it, is_new] = block_diagonal.try_emplace(idx);
    if (is_new) {
      it->second = diagonal.segment(block_size * idx, block_size).asDiagonal();
    } else {
      it->second += diagonal.segment(block_size * idx, block_size).asDiagonal();
    }
  }
}

// sum up diagonal blocks in hash map for parallel reduction
template <typename Scalar>
class BlockDiagonalAccumulator {
 public:
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  inline void add(size_t idx, MatX&& block) {
    auto [it, is_new] = block_diagonal_.try_emplace(idx);
    if (is_new) {
      it->second = std::move(block);
    } else {
      it->second += block;
    }
  }

  inline void add_diag(size_t num_blocks, size_t block_size,
                       const VecX& diagonal) {
    add_diagonal(block_diagonal_, num_blocks, block_size, diagonal);
  }

  inline void join(BlockDiagonalAccumulator& b) {
    for (auto& [k, v] : b.block_diagonal_) {
      auto [it, is_new] = block_diagonal_.try_emplace(k);
      if (is_new) {
        it->second = std::move(v);
      } else {
        it->second += v;
      }
    }
  }

  IndexedBlocks<Scalar> block_diagonal_;
};

}  // namespace basalt
