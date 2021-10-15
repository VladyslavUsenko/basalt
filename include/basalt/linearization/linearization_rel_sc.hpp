#pragma once

#include <basalt/linearization/linearization_base.hpp>

#include <basalt/optimization/accumulator.h>
#include <basalt/vi_estimator/landmark_database.h>
#include <basalt/vi_estimator/sqrt_ba_base.h>
#include <basalt/linearization/landmark_block.hpp>
#include <basalt/utils/cast_utils.hpp>
#include <basalt/utils/time_utils.hpp>

namespace basalt {

template <class Scalar>
class ImuBlock;

template <typename Scalar_, int POSE_SIZE_>
class LinearizationRelSC : public LinearizationBase<Scalar_, POSE_SIZE_> {
 public:
  using Scalar = Scalar_;
  static constexpr int POSE_SIZE = POSE_SIZE_;
  using Base = LinearizationBase<Scalar, POSE_SIZE>;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using VecP = Eigen::Matrix<Scalar, POSE_SIZE, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat36 = Eigen::Matrix<Scalar, 3, 6>;
  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
  using Mat6 = Eigen::Matrix<Scalar, 6, 6>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  using LandmarkBlockPtr = std::unique_ptr<LandmarkBlock<Scalar>>;
  using ImuBlockPtr = std::unique_ptr<ImuBlock<Scalar>>;

  using RelLinData = typename ScBundleAdjustmentBase<Scalar>::RelLinData;

  using typename Base::Options;

  LinearizationRelSC(
      BundleAdjustmentBase<Scalar>* estimator, const AbsOrderMap& aom,
      const Options& options,
      const MargLinData<Scalar>* marg_lin_data = nullptr,
      const ImuLinData<Scalar>* imu_lin_data = nullptr,
      const std::set<FrameId>* used_frames = nullptr,
      const std::unordered_set<KeypointId>* lost_landmarks = nullptr,
      int64_t last_state_to_marg = std::numeric_limits<int64_t>::max());

  // destructor defined in cpp b/c of unique_ptr destructor (ImuBlock)
  // copy/move constructor/assignment-operator for rule-of-five
  ~LinearizationRelSC();
  LinearizationRelSC(const LinearizationRelSC&) = default;
  LinearizationRelSC(LinearizationRelSC&&) = default;
  LinearizationRelSC& operator=(const LinearizationRelSC&) = default;
  LinearizationRelSC& operator=(LinearizationRelSC&&) = default;

  void log_problem_stats(ExecutionStats& stats) const override;

  Scalar linearizeProblem(bool* numerically_valid = nullptr) override;

  void performQR() override;

  void setPoseDamping(const Scalar lambda);

  bool hasPoseDamping() const { return pose_damping_diagonal > 0; }

  Scalar backSubstitute(const VecX& pose_inc) override;

  VecX getJp_diag2() const;

  void scaleJl_cols();

  void scaleJp_cols(const VecX& jacobian_scaling);

  void setLandmarkDamping(Scalar lambda);

  void get_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r) const override;

  void get_dense_H_b(MatX& H, VecX& b) const override;

 protected:  // types
  using PoseLinMapType =
      Eigen::aligned_unordered_map<std::pair<TimeCamId, TimeCamId>,
                                   RelPoseLin<Scalar>>;
  using PoseLinMapTypeConstIter = typename PoseLinMapType::const_iterator;

  using HostLandmarkMapType =
      std::unordered_map<TimeCamId, std::vector<const LandmarkBlock<Scalar>*>>;

 protected:  //  helper
  void get_dense_Q2Jp_Q2r_pose_damping(MatX& Q2Jp, size_t start_idx) const;

  void get_dense_Q2Jp_Q2r_marg_prior(MatX& Q2Jp, VecX& Q2r,
                                     size_t start_idx) const;

  void add_dense_H_b_pose_damping(MatX& H) const;

  void add_dense_H_b_marg_prior(MatX& H, VecX& b) const;

  void add_dense_H_b_imu(DenseAccumulator<Scalar>& accum) const;

  void add_dense_H_b_imu(MatX& H, VecX& b) const;

  // Transform to abs
  static void accumulate_dense_H_b_rel_to_abs(
      const MatX& rel_H, const VecX& rel_b,
      const std::vector<PoseLinMapTypeConstIter>& rpph, const AbsOrderMap& aom,
      DenseAccumulator<Scalar>& accum);

 protected:
  Options options_;

  std::vector<ImuBlockPtr> imu_blocks;

  const BundleAdjustmentBase<Scalar>* estimator;

  LandmarkDatabase<Scalar>& lmdb_;

  const Calibration<Scalar>& calib;

  const AbsOrderMap& aom;
  const std::set<FrameId>* used_frames;

  const MargLinData<Scalar>* marg_lin_data;
  const ImuLinData<Scalar>* imu_lin_data;
  const std::unordered_set<KeypointId>* lost_landmarks;
  int64_t last_state_to_marg;

  Eigen::aligned_vector<RelLinData> rld_vec;

  Scalar pose_damping_diagonal;
  Scalar pose_damping_diagonal_sqrt;

  VecX marg_scaling;

  size_t num_rows_Q2r;
};

}  // namespace basalt
