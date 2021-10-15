#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <basalt/optimization/accumulator.h>
#include <basalt/vi_estimator/landmark_database.h>
#include <basalt/linearization/block_diagonal.hpp>

namespace basalt {

template <class Scalar>
struct RelPoseLin {
  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
  using Mat6 = Eigen::Matrix<Scalar, 6, 6>;

  Mat4 T_t_h;
  Mat6 d_rel_d_h;
  Mat6 d_rel_d_t;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename Scalar>
class LandmarkBlock {
 public:
  struct Options {
    // use Householder instead of Givens for marginalization
    bool use_householder = true;

    // use_valid_projections_only: if true, set invalid projection's
    // residual and jacobian to 0; invalid means z <= 0
    bool use_valid_projections_only = true;

    // if > 0, use huber norm with given threshold, else squared norm
    Scalar huber_parameter = 0;

    // Standard deviation of reprojection error to weight visual measurements
    Scalar obs_std_dev = 1;

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Scalar jacobi_scaling_eps = 1e-6;
  };

  enum State {
    Uninitialized = 0,
    Allocated,
    NumericalFailure,
    Linearized,
    Marginalized
  };

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMatX =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  using RowMat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

  virtual ~LandmarkBlock(){};

  virtual bool isNumericalFailure() const = 0;
  virtual void allocateLandmark(
      Keypoint<Scalar>& lm,
      const Eigen::aligned_unordered_map<std::pair<TimeCamId, TimeCamId>,
                                         RelPoseLin<Scalar>>& relative_pose_lin,
      const Calibration<Scalar>& calib, const AbsOrderMap& aom,
      const Options& options,
      const std::map<TimeCamId, size_t>* rel_order = nullptr) = 0;

  // may set state to NumericalFailure --> linearization at this state is
  // unusable. Numeric check is only performed for residuals that were
  // considered to be used (valid), which depends on
  // use_valid_projections_only setting.
  virtual Scalar linearizeLandmark() = 0;
  virtual void performQR() = 0;

  // Sets damping and maintains upper triangular matrix for landmarks.
  virtual void setLandmarkDamping(Scalar lambda) = 0;

  // lambda < 0 means computing exact model cost change
  virtual void backSubstitute(const VecX& pose_inc, Scalar& l_diff) = 0;

  virtual void addQ2JpTQ2Jp_mult_x(VecX& res, const VecX& x_pose) const = 0;

  virtual void addQ2JpTQ2r(VecX& res) const = 0;

  virtual void addJp_diag2(VecX& res) const = 0;

  virtual void addQ2JpTQ2Jp_blockdiag(
      BlockDiagonalAccumulator<Scalar>& accu) const = 0;

  virtual void scaleJl_cols() = 0;
  virtual void scaleJp_cols(const VecX& jacobian_scaling) = 0;
  virtual void printStorage(const std::string& filename) const = 0;
  virtual State getState() const = 0;

  virtual size_t numReducedCams() const = 0;

  virtual void get_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r,
                                  size_t start_idx) const = 0;

  virtual void get_dense_Q2Jp_Q2r_rel(
      MatX& Q2Jp, VecX& Q2r, size_t start_idx,
      const std::map<TimeCamId, size_t>& rel_order) const = 0;

  virtual void add_dense_H_b(DenseAccumulator<Scalar>& accum) const = 0;

  virtual void add_dense_H_b(MatX& H, VecX& b) const = 0;

  virtual void add_dense_H_b_rel(
      MatX& H_rel, VecX& b_rel,
      const std::map<TimeCamId, size_t>& rel_order) const = 0;

  virtual const Eigen::PermutationMatrix<Eigen::Dynamic>& get_rel_permutation()
      const = 0;

  virtual Eigen::PermutationMatrix<Eigen::Dynamic> compute_rel_permutation(
      const std::map<TimeCamId, size_t>& rel_order) const = 0;

  virtual void add_dense_H_b_rel_2(MatX& H_rel, VecX& b_rel) const = 0;

  virtual TimeCamId getHostKf() const = 0;

  virtual size_t numQ2rows() const = 0;

  // factory method
  template <int POSE_SIZE>
  static std::unique_ptr<LandmarkBlock<Scalar>> createLandmarkBlock();
};

}  // namespace basalt
