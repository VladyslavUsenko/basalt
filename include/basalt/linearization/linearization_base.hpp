#pragma once

#include <Eigen/Dense>

#include <basalt/vi_estimator/ba_base.h>
#include <basalt/linearization/landmark_block.hpp>
#include <basalt/utils/time_utils.hpp>

namespace basalt {

template <typename Scalar_, int POSE_SIZE_>
class LinearizationBase {
 public:
  using Scalar = Scalar_;
  static constexpr int POSE_SIZE = POSE_SIZE_;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  struct Options {
    typename LandmarkBlock<Scalar>::Options lb_options;
    LinearizationType linearization_type;
  };

  virtual ~LinearizationBase() = default;

  virtual void log_problem_stats(ExecutionStats& stats) const = 0;

  virtual Scalar linearizeProblem(bool* numerically_valid = nullptr) = 0;

  virtual void performQR() = 0;

  // virtual void setPoseDamping(const Scalar lambda) = 0;

  // virtual bool hasPoseDamping() const = 0;

  virtual Scalar backSubstitute(const VecX& pose_inc) = 0;

  // virtual VecX getJp_diag2() const = 0;

  // virtual void scaleJl_cols() = 0;

  // virtual void scaleJp_cols(const VecX& jacobian_scaling) = 0;

  // virtual void setLandmarkDamping(Scalar lambda) = 0;

  virtual void get_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r) const = 0;

  virtual void get_dense_H_b(MatX& H, VecX& b) const = 0;

  static std::unique_ptr<LinearizationBase> create(
      BundleAdjustmentBase<Scalar>* estimator, const AbsOrderMap& aom,
      const Options& options,
      const MargLinData<Scalar>* marg_lin_data = nullptr,
      const ImuLinData<Scalar>* imu_lin_data = nullptr,
      const std::set<FrameId>* used_frames = nullptr,
      const std::unordered_set<KeypointId>* lost_landmarks = nullptr,
      int64_t last_state_to_marg = std::numeric_limits<int64_t>::max());
};

bool isLinearizationSqrt(const LinearizationType& type);

}  // namespace basalt
