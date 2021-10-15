#pragma once

#include <fstream>
#include <mutex>

#include <basalt/utils/ba_utils.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <basalt/linearization/landmark_block.hpp>
#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

template <typename Scalar, int POSE_SIZE>
class LandmarkBlockAbsDynamic : public LandmarkBlock<Scalar> {
 public:
  using Options = typename LandmarkBlock<Scalar>::Options;
  using State = typename LandmarkBlock<Scalar>::State;

  inline bool isNumericalFailure() const override {
    return state == State::NumericalFailure;
  }

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMatX =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  virtual inline void allocateLandmark(
      Keypoint<Scalar>& lm,
      const Eigen::aligned_unordered_map<std::pair<TimeCamId, TimeCamId>,
                                         RelPoseLin<Scalar>>& relative_pose_lin,
      const Calibration<Scalar>& calib, const AbsOrderMap& aom,
      const Options& options,
      const std::map<TimeCamId, size_t>* rel_order = nullptr) override {
    // some of the logic assumes the members are at their initial values
    BASALT_ASSERT(state == State::Uninitialized);

    UNUSED(rel_order);

    lm_ptr = &lm;
    options_ = &options;
    calib_ = &calib;

    // TODO: consider for VIO that we have a lot of 0 columns if we just use aom
    // --> add option to use different AOM with reduced size and/or just
    // involved poses --> when accumulating results, check which case we have;
    // if both aom are identical, we don't have to do block-wise operations.
    aom_ = &aom;

    pose_lin_vec.clear();
    pose_lin_vec.reserve(lm.obs.size());
    pose_tcid_vec.clear();
    pose_tcid_vec.reserve(lm.obs.size());

    // LMBs without host frame should not be created
    BASALT_ASSERT(aom.abs_order_map.count(lm.host_kf_id.frame_id) > 0);

    for (const auto& [tcid_t, pos] : lm.obs) {
      size_t i = pose_lin_vec.size();

      auto it = relative_pose_lin.find(std::make_pair(lm.host_kf_id, tcid_t));
      BASALT_ASSERT(it != relative_pose_lin.end());

      if (aom.abs_order_map.count(tcid_t.frame_id) > 0) {
        pose_lin_vec.push_back(&it->second);
      } else {
        // Observation droped for marginalization
        pose_lin_vec.push_back(nullptr);
      }
      pose_tcid_vec.push_back(&it->first);

      res_idx_by_abs_pose_[it->first.first.frame_id].insert(i);   // host
      res_idx_by_abs_pose_[it->first.second.frame_id].insert(i);  // target
    }

    // number of pose-jacobian columns is determined by oam
    padding_idx = aom_->total_size;

    num_rows = pose_lin_vec.size() * 2 + 3;  // residuals and lm damping

    size_t pad = padding_idx % 4;
    if (pad != 0) {
      padding_size = 4 - pad;
    }

    lm_idx = padding_idx + padding_size;
    res_idx = lm_idx + 3;
    num_cols = res_idx + 1;

    // number of columns should now be multiple of 4 for good memory alignment
    // TODO: test extending this to 8 --> 32byte alignment for float?
    BASALT_ASSERT(num_cols % 4 == 0);

    storage.resize(num_rows, num_cols);

    damping_rotations.clear();
    damping_rotations.reserve(6);

    state = State::Allocated;
  }

  // may set state to NumericalFailure --> linearization at this state is
  // unusable. Numeric check is only performed for residuals that were
  // considered to be used (valid), which depends on
  // use_valid_projections_only setting.
  virtual inline Scalar linearizeLandmark() override {
    BASALT_ASSERT(state == State::Allocated ||
                  state == State::NumericalFailure ||
                  state == State::Linearized || state == State::Marginalized);

    // storage.setZero(num_rows, num_cols);
    storage.setZero();
    damping_rotations.clear();
    damping_rotations.reserve(6);

    bool numerically_valid = true;

    Scalar error_sum = 0;

    size_t i = 0;
    for (const auto& [tcid_t, obs] : lm_ptr->obs) {
      std::visit(
          [&, obs = obs](const auto& cam) {
            // TODO: The pose_lin_vec[i] == nullptr is intended to deal with
            // dropped measurements during marginalization. However, dropped
            // measurements should only occur for the remaining frames, not for
            // the marginalized frames. Maybe these are observations bewtween
            // two marginalized frames, if more than one is marginalized at the
            // same time? But those we would not have to drop... Double check if
            // and when this happens and possibly resolve by fixing handling
            // here, or else updating the measurements in lmdb before calling
            // linearization. Otherwise, check where else we need a `if
            // (pose_lin_vec[i])` check or `pose_lin_vec[i] != nullptr` assert
            // in this class.

            if (pose_lin_vec[i]) {
              size_t obs_idx = i * 2;
              size_t abs_h_idx =
                  aom_->abs_order_map.at(pose_tcid_vec[i]->first.frame_id)
                      .first;
              size_t abs_t_idx =
                  aom_->abs_order_map.at(pose_tcid_vec[i]->second.frame_id)
                      .first;

              Vec2 res;
              Eigen::Matrix<Scalar, 2, POSE_SIZE> d_res_d_xi;
              Eigen::Matrix<Scalar, 2, 3> d_res_d_p;

              using CamT = std::decay_t<decltype(cam)>;
              bool valid = linearizePoint<Scalar, CamT>(
                  obs, *lm_ptr, pose_lin_vec[i]->T_t_h, cam, res, &d_res_d_xi,
                  &d_res_d_p);

              if (!options_->use_valid_projections_only || valid) {
                numerically_valid = numerically_valid &&
                                    d_res_d_xi.array().isFinite().all() &&
                                    d_res_d_p.array().isFinite().all();

                const Scalar res_squared = res.squaredNorm();
                const auto [weighted_error, weight] =
                    compute_error_weight(res_squared);
                const Scalar sqrt_weight =
                    std::sqrt(weight) / options_->obs_std_dev;

                error_sum += weighted_error /
                             (options_->obs_std_dev * options_->obs_std_dev);

                storage.template block<2, 3>(obs_idx, lm_idx) =
                    sqrt_weight * d_res_d_p;
                storage.template block<2, 1>(obs_idx, res_idx) =
                    sqrt_weight * res;

                d_res_d_xi *= sqrt_weight;
                storage.template block<2, 6>(obs_idx, abs_h_idx) +=
                    d_res_d_xi * pose_lin_vec[i]->d_rel_d_h;
                storage.template block<2, 6>(obs_idx, abs_t_idx) +=
                    d_res_d_xi * pose_lin_vec[i]->d_rel_d_t;
              }
            }

            i++;
          },
          calib_->intrinsics[tcid_t.cam_id].variant);
    }

    if (numerically_valid) {
      state = State::Linearized;
    } else {
      state = State::NumericalFailure;
    }

    return error_sum;
  }

  virtual inline void performQR() override {
    BASALT_ASSERT(state == State::Linearized);

    // Since we use dense matrices Householder QR might be better:
    // https://mathoverflow.net/questions/227543/why-householder-reflection-is-better-than-givens-rotation-in-dense-linear-algebr

    if (options_->use_householder) {
      performQRHouseholder();
    } else {
      performQRGivens();
    }

    state = State::Marginalized;
  }

  // Sets damping and maintains upper triangular matrix for landmarks.
  virtual inline void setLandmarkDamping(Scalar lambda) override {
    BASALT_ASSERT(state == State::Marginalized);
    BASALT_ASSERT(lambda >= 0);

    if (hasLandmarkDamping()) {
      BASALT_ASSERT(damping_rotations.size() == 6);

      // undo dampening
      for (int n = 2; n >= 0; n--) {
        for (int m = n; m >= 0; m--) {
          storage.applyOnTheLeft(num_rows - 3 + n - m, n,
                                 damping_rotations.back().adjoint());
          damping_rotations.pop_back();
        }
      }
    }

    if (lambda == 0) {
      storage.template block<3, 3>(num_rows - 3, lm_idx).diagonal().setZero();
    } else {
      BASALT_ASSERT(Jl_col_scale.array().isFinite().all());

      storage.template block<3, 3>(num_rows - 3, lm_idx)
          .diagonal()
          .setConstant(sqrt(lambda));

      BASALT_ASSERT(damping_rotations.empty());

      // apply dampening and remember rotations to undo
      for (int n = 0; n < 3; n++) {
        for (int m = 0; m <= n; m++) {
          damping_rotations.emplace_back();
          damping_rotations.back().makeGivens(
              storage(n, lm_idx + n),
              storage(num_rows - 3 + n - m, lm_idx + n));
          storage.applyOnTheLeft(num_rows - 3 + n - m, n,
                                 damping_rotations.back());
        }
      }
    }
  }

  // lambda < 0 means computing exact model cost change
  virtual inline void backSubstitute(const VecX& pose_inc,
                                     Scalar& l_diff) override {
    BASALT_ASSERT(state == State::Marginalized);

    // For now we include all columns in LMB
    BASALT_ASSERT(pose_inc.size() == signed_cast(padding_idx));

    const auto Q1Jl = storage.template block<3, 3>(0, lm_idx)
                          .template triangularView<Eigen::Upper>();

    const auto Q1Jr = storage.col(res_idx).template head<3>();
    const auto Q1Jp = storage.topLeftCorner(3, padding_idx);

    Vec3 inc = -Q1Jl.solve(Q1Jr + Q1Jp * pose_inc);

    // We want to compute the model cost change. The model function is
    //
    //     L(inc) = F(x) + inc^T J^T r + 0.5 inc^T J^T J inc
    //
    // and thus the expected decrease in cost for the computed increment is
    //
    //     l_diff = L(0) - L(inc)
    //            = - inc^T J^T r - 0.5 inc^T J^T J inc
    //            = - inc^T J^T (r + 0.5 J inc)
    //            = - (J inc)^T (r + 0.5 (J inc)).
    //
    // Here we have J = [Jp, Jl] under the orthogonal projection Q = [Q1, Q2],
    // i.e. the linearized system (model cost) is
    //
    //    L(inc) = 0.5 || J inc + r ||^2 = 0.5 || Q^T J inc + Q^T r ||^2
    //
    // and below we thus compute
    //
    //    l_diff = - (Q^T J inc)^T (Q^T r + 0.5 (Q^T J inc)).
    //
    // We have
    //             | Q1^T |            | Q1^T Jp   Q1^T Jl |
    //    Q^T J =  |      | [Jp, Jl] = |                   |
    //             | Q2^T |            | Q2^T Jp      0    |.
    //
    // Note that Q2 is the nullspace of Jl, and Q1^T Jl == R. So with inc =
    // [incp^T, incl^T]^T we have
    //
    //                | Q1^T Jp incp + Q1^T Jl incl |
    //    Q^T J inc = |                             |
    //                | Q2^T Jp incp                |
    //

    // undo damping before we compute the model cost difference
    setLandmarkDamping(0);

    // compute "Q^T J incp"
    VecX QJinc = storage.topLeftCorner(num_rows - 3, padding_idx) * pose_inc;

    // add "Q1^T Jl incl" to the first 3 rows
    QJinc.template head<3>() += Q1Jl * inc;

    auto Qr = storage.col(res_idx).head(num_rows - 3);
    l_diff -= QJinc.transpose() * (Scalar(0.5) * QJinc + Qr);

    // TODO: detect and handle case like ceres, allowing a few iterations but
    // stopping eventually
    if (!inc.array().isFinite().all() ||
        !lm_ptr->direction.array().isFinite().all() ||
        !std::isfinite(lm_ptr->inv_dist)) {
      std::cerr << "Numerical failure in backsubstitution\n";
    }

    // Note: scale only after computing model cost change
    inc.array() *= Jl_col_scale.array();

    lm_ptr->direction += inc.template head<2>();
    lm_ptr->inv_dist = std::max(Scalar(0), lm_ptr->inv_dist + inc[2]);
  }

  virtual inline size_t numReducedCams() const override {
    BASALT_LOG_FATAL("check what we mean by numReducedCams for absolute poses");
    return pose_lin_vec.size();
  }

  inline void addQ2JpTQ2Jp_mult_x(VecX& res,
                                  const VecX& x_pose) const override {
    UNUSED(res);
    UNUSED(x_pose);
    BASALT_LOG_FATAL("not implemented");
  }

  virtual inline void addQ2JpTQ2r(VecX& res) const override {
    UNUSED(res);
    BASALT_LOG_FATAL("not implemented");
  }

  virtual inline void addJp_diag2(VecX& res) const override {
    BASALT_ASSERT(state == State::Linearized);

    for (const auto& [frame_id, idx_set] : res_idx_by_abs_pose_) {
      const int pose_idx = aom_->abs_order_map.at(frame_id).first;
      for (const int i : idx_set) {
        const auto block = storage.block(2 * i, pose_idx, 2, POSE_SIZE);

        res.template segment<POSE_SIZE>(pose_idx) +=
            block.colwise().squaredNorm();
      }
    }
  }

  virtual inline void addQ2JpTQ2Jp_blockdiag(
      BlockDiagonalAccumulator<Scalar>& accu) const override {
    UNUSED(accu);
    BASALT_LOG_FATAL("not implemented");
  }

  virtual inline void scaleJl_cols() override {
    BASALT_ASSERT(state == State::Linearized);

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Jl_col_scale =
        (options_->jacobi_scaling_eps +
         storage.block(0, lm_idx, num_rows - 3, 3).colwise().norm().array())
            .inverse();

    storage.block(0, lm_idx, num_rows - 3, 3) *= Jl_col_scale.asDiagonal();
  }

  virtual inline void scaleJp_cols(const VecX& jacobian_scaling) override {
    BASALT_ASSERT(state == State::Marginalized);

    // we assume we apply scaling before damping (we exclude the last 3 rows)
    BASALT_ASSERT(!hasLandmarkDamping());

    storage.topLeftCorner(num_rows - 3, padding_idx) *=
        jacobian_scaling.asDiagonal();
  }

  inline bool hasLandmarkDamping() const { return !damping_rotations.empty(); }

  virtual inline void printStorage(const std::string& filename) const override {
    std::ofstream f(filename);

    Eigen::IOFormat CleanFmt(4, 0, " ", "\n", "", "");

    f << "Storage (state: " << state
      << ", damping: " << (hasLandmarkDamping() ? "yes" : "no")
      << " Jl_col_scale: " << Jl_col_scale.transpose() << "):\n"
      << storage.format(CleanFmt) << std::endl;

    f.close();
  }
#if 0
  virtual inline void stage2(
      Scalar lambda, const VecX* jacobian_scaling, VecX* precond_diagonal2,
      BlockDiagonalAccumulator<Scalar>* precond_block_diagonal,
      VecX& bref) override {
    // 1. scale jacobian
    if (jacobian_scaling) {
      scaleJp_cols(*jacobian_scaling);
    }

    // 2. dampen landmarks
    setLandmarkDamping(lambda);

    // 3a. compute diagonal preconditioner (SCHUR_JACOBI_DIAGONAL)
    if (precond_diagonal2) {
      addQ2Jp_diag2(*precond_diagonal2);
    }

    // 3b. compute block diagonal preconditioner (SCHUR_JACOBI)
    if (precond_block_diagonal) {
      addQ2JpTQ2Jp_blockdiag(*precond_block_diagonal);
    }

    // 4. compute rhs of reduced camera normal equations
    addQ2JpTQ2r(bref);
  }
#endif

  inline State getState() const override { return state; }

  virtual inline size_t numQ2rows() const override { return num_rows - 3; }

 protected:
  inline void performQRGivens() {
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    Eigen::JacobiRotation<Scalar> gr;
    for (size_t n = 0; n < 3; n++) {
      for (size_t m = num_rows - 4; m > n; m--) {
        gr.makeGivens(storage(m - 1, lm_idx + n), storage(m, lm_idx + n));
        storage.applyOnTheLeft(m, m - 1, gr);
      }
    }
  }

  inline void performQRHouseholder() {
    VecX tempVector1(num_cols);
    VecX tempVector2(num_rows - 3);

    for (size_t k = 0; k < 3; ++k) {
      size_t remainingRows = num_rows - k - 3;

      Scalar beta;
      Scalar tau;
      storage.col(lm_idx + k)
          .segment(k, remainingRows)
          .makeHouseholder(tempVector2, tau, beta);

      storage.block(k, 0, remainingRows, num_cols)
          .applyHouseholderOnTheLeft(tempVector2, tau, tempVector1.data());
    }
  }

  inline std::tuple<Scalar, Scalar> compute_error_weight(
      Scalar res_squared) const {
    // Note: Definition of cost is 0.5 ||r(x)||^2 to be in line with ceres

    if (options_->huber_parameter > 0) {
      // use huber norm
      const Scalar huber_weight =
          res_squared <= options_->huber_parameter * options_->huber_parameter
              ? Scalar(1)
              : options_->huber_parameter / std::sqrt(res_squared);
      const Scalar error =
          Scalar(0.5) * (2 - huber_weight) * huber_weight * res_squared;
      return {error, huber_weight};
    } else {
      // use squared norm
      return {Scalar(0.5) * res_squared, Scalar(1)};
    }
  }

  void get_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r,
                          size_t start_idx) const override {
    Q2r.segment(start_idx, num_rows - 3) =
        storage.col(res_idx).tail(num_rows - 3);

    BASALT_ASSERT(Q2Jp.cols() == signed_cast(padding_idx));

    Q2Jp.block(start_idx, 0, num_rows - 3, padding_idx) =
        storage.block(3, 0, num_rows - 3, padding_idx);
  }

  void get_dense_Q2Jp_Q2r_rel(
      MatX& Q2Jp, VecX& Q2r, size_t start_idx,
      const std::map<TimeCamId, size_t>& rel_order) const override {
    UNUSED(Q2Jp);
    UNUSED(Q2r);
    UNUSED(start_idx);
    UNUSED(rel_order);
    BASALT_LOG_FATAL("not implemented");
  }

  void add_dense_H_b(DenseAccumulator<Scalar>& accum) const override {
    UNUSED(accum);
    BASALT_LOG_FATAL("not implemented");
  }

  void add_dense_H_b(MatX& H, VecX& b) const override {
    const auto r = storage.col(res_idx).tail(num_rows - 3);
    const auto J = storage.block(3, 0, num_rows - 3, padding_idx);

    H.noalias() += J.transpose() * J;
    b.noalias() += J.transpose() * r;
  }

  void add_dense_H_b_rel(
      MatX& H_rel, VecX& b_rel,
      const std::map<TimeCamId, size_t>& rel_order) const override {
    UNUSED(H_rel);
    UNUSED(b_rel);
    UNUSED(rel_order);
    BASALT_LOG_FATAL("not implemented");
  }

  const Eigen::PermutationMatrix<Eigen::Dynamic>& get_rel_permutation()
      const override {
    BASALT_LOG_FATAL("not implemented");
  }

  Eigen::PermutationMatrix<Eigen::Dynamic> compute_rel_permutation(
      const std::map<TimeCamId, size_t>& rel_order) const override {
    UNUSED(rel_order);
    BASALT_LOG_FATAL("not implemented");
  }

  void add_dense_H_b_rel_2(MatX& H_rel, VecX& b_rel) const override {
    UNUSED(H_rel);
    UNUSED(b_rel);
    BASALT_LOG_FATAL("not implemented");
  }

  virtual TimeCamId getHostKf() const override { return lm_ptr->host_kf_id; }

 private:
  // Dense storage for pose Jacobians, padding, landmark Jacobians and
  // residuals [J_p | pad | J_l | res]
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      storage;

  Vec3 Jl_col_scale = Vec3::Ones();
  std::vector<Eigen::JacobiRotation<Scalar>> damping_rotations;

  std::vector<const RelPoseLin<Scalar>*> pose_lin_vec;
  std::vector<const std::pair<TimeCamId, TimeCamId>*> pose_tcid_vec;
  size_t padding_idx = 0;
  size_t padding_size = 0;
  size_t lm_idx = 0;
  size_t res_idx = 0;

  size_t num_cols = 0;
  size_t num_rows = 0;

  const Options* options_ = nullptr;

  State state = State::Uninitialized;

  Keypoint<Scalar>* lm_ptr = nullptr;
  const Calibration<Scalar>* calib_ = nullptr;
  const AbsOrderMap* aom_ = nullptr;

  std::map<int64_t, std::set<int>> res_idx_by_abs_pose_;
};

}  // namespace basalt
