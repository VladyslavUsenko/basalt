#pragma once

#include <basalt/imu/preintegration.h>
#include <basalt/optimization/accumulator.h>
#include <basalt/utils/imu_types.h>

namespace basalt {

template <class Scalar_>
class ImuBlock {
 public:
  using Scalar = Scalar_;

  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  ImuBlock(const IntegratedImuMeasurement<Scalar>* meas,
           const ImuLinData<Scalar>* imu_lin_data, const AbsOrderMap& aom)
      : imu_meas(meas), imu_lin_data(imu_lin_data), aom(aom) {
    Jp.resize(POSE_VEL_BIAS_SIZE, 2 * POSE_VEL_BIAS_SIZE);
    r.resize(POSE_VEL_BIAS_SIZE);
  }

  Scalar linearizeImu(
      const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
          frame_states) {
    Jp.setZero();
    r.setZero();

    const int64_t start_t = imu_meas->get_start_t_ns();
    const int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = 0;
    const size_t end_idx = POSE_VEL_BIAS_SIZE;

    PoseVelBiasStateWithLin<Scalar> start_state = frame_states.at(start_t);
    PoseVelBiasStateWithLin<Scalar> end_state = frame_states.at(end_t);

    typename IntegratedImuMeasurement<Scalar>::MatNN d_res_d_start, d_res_d_end;
    typename IntegratedImuMeasurement<Scalar>::MatN3 d_res_d_bg, d_res_d_ba;

    typename PoseVelState<Scalar>::VecN res = imu_meas->residual(
        start_state.getStateLin(), imu_lin_data->g, end_state.getStateLin(),
        start_state.getStateLin().bias_gyro,
        start_state.getStateLin().bias_accel, &d_res_d_start, &d_res_d_end,
        &d_res_d_bg, &d_res_d_ba);

    if (start_state.isLinearized() || end_state.isLinearized()) {
      res = imu_meas->residual(
          start_state.getState(), imu_lin_data->g, end_state.getState(),
          start_state.getState().bias_gyro, start_state.getState().bias_accel);
    }

    // error
    Scalar imu_error =
        Scalar(0.5) * (imu_meas->get_sqrt_cov_inv() * res).squaredNorm();

    // imu residual linearization
    Jp.template block<9, 9>(0, start_idx) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_start;
    Jp.template block<9, 9>(0, end_idx) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_end;

    Jp.template block<9, 3>(0, start_idx + 9) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_bg;
    Jp.template block<9, 3>(0, start_idx + 12) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_ba;

    r.template segment<9>(0) = imu_meas->get_sqrt_cov_inv() * res;

    // difference between biases
    Scalar dt = imu_meas->get_dt_ns() * Scalar(1e-9);

    Vec3 gyro_bias_weight_dt =
        imu_lin_data->gyro_bias_weight_sqrt / std::sqrt(dt);
    Vec3 res_bg =
        start_state.getState().bias_gyro - end_state.getState().bias_gyro;

    Jp.template block<3, 3>(9, start_idx + 9) =
        gyro_bias_weight_dt.asDiagonal();
    Jp.template block<3, 3>(9, end_idx + 9) =
        (-gyro_bias_weight_dt).asDiagonal();

    r.template segment<3>(9) += gyro_bias_weight_dt.asDiagonal() * res_bg;

    Scalar bg_error =
        Scalar(0.5) * (gyro_bias_weight_dt.asDiagonal() * res_bg).squaredNorm();

    Vec3 accel_bias_weight_dt =
        imu_lin_data->accel_bias_weight_sqrt / std::sqrt(dt);
    Vec3 res_ba =
        start_state.getState().bias_accel - end_state.getState().bias_accel;

    Jp.template block<3, 3>(12, start_idx + 12) =
        accel_bias_weight_dt.asDiagonal();
    Jp.template block<3, 3>(12, end_idx + 12) =
        (-accel_bias_weight_dt).asDiagonal();

    r.template segment<3>(12) += accel_bias_weight_dt.asDiagonal() * res_ba;

    Scalar ba_error =
        Scalar(0.5) *
        (accel_bias_weight_dt.asDiagonal() * res_ba).squaredNorm();

    return imu_error + bg_error + ba_error;
  }

  void add_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r, size_t row_start_idx) const {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    Q2Jp.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(row_start_idx,
                                                                start_idx) +=
        Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>();

    Q2Jp.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(row_start_idx,
                                                                end_idx) +=
        Jp.template topRightCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>();

    Q2r.template segment<POSE_VEL_BIAS_SIZE>(row_start_idx) += r;
  }

  void add_dense_H_b(DenseAccumulator<Scalar>& accum) const {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    const MatX H = Jp.transpose() * Jp;
    const VecX b = Jp.transpose() * r;

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        start_idx, start_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(0, 0));

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        end_idx, start_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
            POSE_VEL_BIAS_SIZE, 0));

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        start_idx, end_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
            0, POSE_VEL_BIAS_SIZE));

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        end_idx, end_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
            POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE));

    accum.template addB<POSE_VEL_BIAS_SIZE>(
        start_idx, b.template segment<POSE_VEL_BIAS_SIZE>(0));
    accum.template addB<POSE_VEL_BIAS_SIZE>(
        end_idx, b.template segment<POSE_VEL_BIAS_SIZE>(POSE_VEL_BIAS_SIZE));
  }

  void scaleJp_cols(const VecX& jacobian_scaling) {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>() *=
        jacobian_scaling.template segment<POSE_VEL_BIAS_SIZE>(start_idx)
            .asDiagonal();

    Jp.template topRightCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>() *=
        jacobian_scaling.template segment<POSE_VEL_BIAS_SIZE>(end_idx)
            .asDiagonal();
  }

  void addJp_diag2(VecX& res) const {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    res.template segment<POSE_VEL_BIAS_SIZE>(start_idx) +=
        Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>()
            .colwise()
            .squaredNorm();

    res.template segment<POSE_VEL_BIAS_SIZE>(end_idx) +=
        Jp.template topRightCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>()
            .colwise()
            .squaredNorm();
  }

  void backSubstitute(const VecX& pose_inc, Scalar& l_diff) {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    VecX pose_inc_reduced(2 * POSE_VEL_BIAS_SIZE);
    pose_inc_reduced.template head<POSE_VEL_BIAS_SIZE>() =
        pose_inc.template segment<POSE_VEL_BIAS_SIZE>(start_idx);
    pose_inc_reduced.template tail<POSE_VEL_BIAS_SIZE>() =
        pose_inc.template segment<POSE_VEL_BIAS_SIZE>(end_idx);

    // We want to compute the model cost change. The model function is
    //
    //     L(inc) = F(x) + incT JT r + 0.5 incT JT J inc
    //
    // and thus the expect decrease in cost for the computed increment is
    //
    //     l_diff = L(0) - L(inc)
    //            = - incT JT r - 0.5 incT JT J inc.
    //            = - incT JT (r + 0.5 J inc)
    //            = - (J inc)T (r + 0.5 (J inc))

    VecX Jinc = Jp * pose_inc_reduced;
    l_diff -= Jinc.transpose() * (Scalar(0.5) * Jinc + r);
  }

 protected:
  std::array<FrameId, 2> frame_ids;
  MatX Jp;
  VecX r;

  const IntegratedImuMeasurement<Scalar>* imu_meas;
  const ImuLinData<Scalar>* imu_lin_data;
  const AbsOrderMap& aom;
};  // namespace basalt

}  // namespace basalt
