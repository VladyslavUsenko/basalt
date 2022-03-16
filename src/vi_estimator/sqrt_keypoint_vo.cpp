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

#include <basalt/vi_estimator/marg_helper.h>
#include <basalt/vi_estimator/sqrt_keypoint_vo.h>

#include <basalt/optimization/accumulator.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/system_utils.h>
#include <basalt/utils/cast_utils.hpp>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

#include <basalt/linearization/linearization_base.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <fmt/format.h>

#include <chrono>

namespace basalt {

template <class Scalar_>
SqrtKeypointVoEstimator<Scalar_>::SqrtKeypointVoEstimator(
    const basalt::Calibration<double>& calib_, const VioConfig& config_)
    : take_kf(true),
      frames_after_kf(0),
      initialized(false),
      config(config_),
      lambda(config_.vio_lm_lambda_initial),
      min_lambda(config_.vio_lm_lambda_min),
      max_lambda(config_.vio_lm_lambda_max),
      lambda_vee(2) {
  obs_std_dev = Scalar(config.vio_obs_std_dev);
  huber_thresh = Scalar(config.vio_obs_huber_thresh);
  calib = calib_.cast<Scalar>();

  // Setup marginalization
  marg_data.is_sqrt = config.vio_sqrt_marg;
  marg_data.H.setZero(POSE_SIZE, POSE_SIZE);
  marg_data.b.setZero(POSE_SIZE);

  // Version without prior
  nullspace_marg_data.is_sqrt = marg_data.is_sqrt;
  nullspace_marg_data.H.setZero(POSE_SIZE, POSE_SIZE);
  nullspace_marg_data.b.setZero(POSE_SIZE);

  // prior on pose
  if (marg_data.is_sqrt) {
    marg_data.H.diagonal().setConstant(
        std::sqrt(Scalar(config.vio_init_pose_weight)));
  } else {
    marg_data.H.diagonal().setConstant(Scalar(config.vio_init_pose_weight));
  }

  std::cout << "marg_H (sqrt:" << marg_data.is_sqrt << ")\n"
            << marg_data.H << std::endl;

  max_states = config.vio_max_states;
  max_kfs = config.vio_max_kfs;

  vision_data_queue.set_capacity(10);
  imu_data_queue.set_capacity(300);
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::initialize(
    int64_t t_ns, const Sophus::SE3d& T_w_i, const Eigen::Vector3d& vel_w_i,
    const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
  UNUSED(vel_w_i);

  initialized = true;
  T_w_i_init = T_w_i.cast<Scalar>();

  last_state_t_ns = t_ns;
  frame_poses[t_ns] = PoseStateWithLin<Scalar>(t_ns, T_w_i_init, true);

  marg_data.order.abs_order_map[t_ns] = std::make_pair(0, POSE_SIZE);
  marg_data.order.total_size = POSE_SIZE;
  marg_data.order.items = 1;

  initialize(bg, ba);
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::initialize(const Eigen::Vector3d& bg,
                                                  const Eigen::Vector3d& ba) {
  UNUSED(bg);
  UNUSED(ba);

  auto proc_func = [&] {
    OpticalFlowResult::Ptr prev_frame, curr_frame;
    bool add_pose = false;

    while (true) {
      // get next optical flow result (blocking if queue empty)
      vision_data_queue.pop(curr_frame);

      if (config.vio_enforce_realtime) {
        // drop current frame if another frame is already in the queue.
        while (!vision_data_queue.empty()) vision_data_queue.pop(curr_frame);
      }

      if (!curr_frame.get()) {
        break;
      }

      // Correct camera time offset (relevant for VIO)
      // curr_frame->t_ns += calib.cam_time_offset_ns;

      // this is VO not VIO --> just drain IMU queue and ignore
      while (!imu_data_queue.empty()) {
        ImuData<double>::Ptr d;
        imu_data_queue.pop(d);
      }

      if (!initialized) {
        last_state_t_ns = curr_frame->t_ns;

        frame_poses[last_state_t_ns] =
            PoseStateWithLin(last_state_t_ns, T_w_i_init, true);

        marg_data.order.abs_order_map[last_state_t_ns] =
            std::make_pair(0, POSE_SIZE);
        marg_data.order.total_size = POSE_SIZE;
        marg_data.order.items = 1;

        nullspace_marg_data.order = marg_data.order;

        std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
        std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;

        if (config.vio_debug || config.vio_extended_logging) {
          logMargNullspace();
        }

        initialized = true;
      }

      if (prev_frame) {
        add_pose = true;
      }

      measure(curr_frame, add_pose);
      prev_frame = curr_frame;
    }

    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_marg_queue) out_marg_queue->push(nullptr);
    if (out_state_queue) out_state_queue->push(nullptr);

    finished = true;

    std::cout << "Finished VIOFilter " << std::endl;
  };

  processing_thread.reset(new std::thread(proc_func));
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::addVisionToQueue(
    const OpticalFlowResult::Ptr& data) {
  vision_data_queue.push(data);
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::addIMUToQueue(
    const ImuData<double>::Ptr& data) {
  UNUSED(data);
}

template <class Scalar_>
bool SqrtKeypointVoEstimator<Scalar_>::measure(
    const OpticalFlowResult::Ptr& opt_flow_meas, const bool add_pose) {
  stats_sums_.add("frame_id", opt_flow_meas->t_ns).format("none");
  Timer t_total;

  //  std::cout << "=== measure frame " << opt_flow_meas->t_ns << "\n";
  //  std::cout.flush();

  // TODO: For VO there is probably no point to non kfs as state in the sliding
  // window... Just do pose-only optimization and never enter them into the
  // sliding window.
  // TODO: If we do pose-only optimization first for all frames (also KFs), this
  // may also speed up the joint optimization.
  // TODO: Moreover, initial pose-only localization may allow us to project
  // untracked landmarks and "rediscover them" by optical flow or feature /
  // patch matching.

  if (add_pose) {
    // The state for the first frame is added by the initialization code, but
    // otherwise we insert a new pose state here. So add_pose is only false
    // right after initialization.

    const PoseStateWithLin<Scalar>& curr_state =
        frame_poses.at(last_state_t_ns);

    last_state_t_ns = opt_flow_meas->t_ns;

    PoseStateWithLin next_state(opt_flow_meas->t_ns, curr_state.getPose());
    frame_poses[last_state_t_ns] = next_state;
  }

  // invariants: opt_flow_meas->t_ns is last pose state and equal to
  // last_state_t_ns
  BASALT_ASSERT(opt_flow_meas->t_ns == last_state_t_ns);
  BASALT_ASSERT(!frame_poses.empty());
  BASALT_ASSERT(last_state_t_ns == frame_poses.rbegin()->first);

  // save results
  prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

  // For feature tracks that exist as landmarks, add the new frames as
  // additional observations. For every host frame, compute how many of it's
  // landmarks are tracked by the current frame: this will help later during
  // marginalization to remove the host frames with a low number of landmarks
  // connected with the latest frame. For new tracks, remember their ids for
  // possible later landmark creation.
  int connected0 = 0;                           // num tracked landmarks cam0
  std::map<int64_t, int> num_points_connected;  // num tracked landmarks by host
  std::unordered_set<int> unconnected_obs0;     // new tracks cam0
  std::vector<std::vector<int>> connected_obs0(
      opt_flow_meas->observations.size());

  for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
    TimeCamId tcid_target(opt_flow_meas->t_ns, i);

    for (const auto& kv_obs : opt_flow_meas->observations[i]) {
      int kpt_id = kv_obs.first;

      if (lmdb.landmarkExists(kpt_id)) {
        const TimeCamId& tcid_host = lmdb.getLandmark(kpt_id).host_kf_id;

        KeypointObservation<Scalar> kobs;
        kobs.kpt_id = kpt_id;
        kobs.pos = kv_obs.second.translation().cast<Scalar>();

        lmdb.addObservation(tcid_target, kobs);
        // obs[tcid_host][tcid_target].push_back(kobs);

        num_points_connected[tcid_host.frame_id]++;
        connected_obs0[i].emplace_back(kpt_id);

        if (i == 0) connected0++;
      } else {
        if (i == 0) {
          unconnected_obs0.emplace(kpt_id);
        }
      }
    }
  }

  if (Scalar(connected0) / (connected0 + unconnected_obs0.size()) <
          Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;

  if (config.vio_debug) {
    std::cout << "connected0 " << connected0 << " unconnected0 "
              << unconnected_obs0.size() << std::endl;
  }

  BundleAdjustmentBase<Scalar>::optimize_single_frame_pose(
      frame_poses[last_state_t_ns], connected_obs0);

  if (take_kf) {
    // For keyframes, we don't only add pose state and observations to existing
    // landmarks (done above for all frames), but also triangulate new
    // landmarks.

    // Triangulate new points from one of the observations (with sufficient
    // baseline) and make keyframe for camera 0
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);

    TimeCamId tcidl(opt_flow_meas->t_ns, 0);

    int num_points_added = 0;
    for (int lm_id : unconnected_obs0) {
      // Find all observations
      std::map<TimeCamId, KeypointObservation<Scalar>> kp_obs;

      for (const auto& kv : prev_opt_flow_res) {
        for (size_t k = 0; k < kv.second->observations.size(); k++) {
          auto it = kv.second->observations[k].find(lm_id);
          if (it != kv.second->observations[k].end()) {
            TimeCamId tcido(kv.first, k);

            KeypointObservation<Scalar> kobs;
            kobs.kpt_id = lm_id;
            kobs.pos = it->second.translation().template cast<Scalar>();

            // obs[tcidl][tcido].push_back(kobs);
            kp_obs[tcido] = kobs;
          }
        }
      }

      // triangulate
      bool valid_kp = false;
      const Scalar min_triang_distance2 =
          Scalar(config.vio_min_triangulation_dist *
                 config.vio_min_triangulation_dist);
      for (const auto& kv_obs : kp_obs) {
        if (valid_kp) break;
        TimeCamId tcido = kv_obs.first;

        const Vec2 p0 = opt_flow_meas->observations.at(0)
                            .at(lm_id)
                            .translation()
                            .cast<Scalar>();
        const Vec2 p1 = prev_opt_flow_res[tcido.frame_id]
                            ->observations[tcido.cam_id]
                            .at(lm_id)
                            .translation()
                            .template cast<Scalar>();

        Vec4 p0_3d, p1_3d;
        bool valid1 = calib.intrinsics[0].unproject(p0, p0_3d);
        bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
        if (!valid1 || !valid2) continue;

        SE3 T_i0_i1 = getPoseStateWithLin(tcidl.frame_id).getPose().inverse() *
                      getPoseStateWithLin(tcido.frame_id).getPose();
        SE3 T_0_1 =
            calib.T_i_c[0].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

        if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

        Vec4 p0_triangulated = triangulate(p0_3d.template head<3>(),
                                           p1_3d.template head<3>(), T_0_1);

        if (p0_triangulated.array().isFinite().all() &&
            p0_triangulated[3] > 0 && p0_triangulated[3] < Scalar(3.0)) {
          Keypoint<Scalar> kpt_pos;
          kpt_pos.host_kf_id = tcidl;
          kpt_pos.direction =
              StereographicParam<Scalar>::project(p0_triangulated);
          kpt_pos.inv_dist = p0_triangulated[3];
          lmdb.addLandmark(lm_id, kpt_pos);

          num_points_added++;
          valid_kp = true;
        }
      }

      if (valid_kp) {
        for (const auto& kv_obs : kp_obs) {
          lmdb.addObservation(kv_obs.first, kv_obs.second);
        }

        // TODO: non-linear refinement of landmark position from all
        // observations; may speed up later joint optimization
      }
    }

    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }

  std::unordered_set<KeypointId> lost_landmaks;
  if (config.vio_marg_lost_landmarks) {
    for (const auto& kv : lmdb.getLandmarks()) {
      bool connected = false;
      for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
        if (opt_flow_meas->observations[i].count(kv.first) > 0)
          connected = true;
      }
      if (!connected) {
        lost_landmaks.emplace(kv.first);
      }
    }
  }

  optimize_and_marg(num_points_connected, lost_landmaks);

  if (out_state_queue) {
    const PoseStateWithLin<Scalar>& p = frame_poses.at(last_state_t_ns);

    typename PoseVelBiasState<double>::Ptr data(new PoseVelBiasState<double>(
        p.getT_ns(), p.getPose().template cast<double>(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero()));

    out_state_queue->push(data);
  }

  if (out_vis_queue) {
    VioVisualizationData::Ptr data(new VioVisualizationData);

    data->t_ns = last_state_t_ns;

    BASALT_ASSERT(frame_states.empty());

    for (const auto& kv : frame_poses) {
      data->frames.emplace_back(kv.second.getPose().template cast<double>());
    }

    get_current_points(data->points, data->point_ids);

    data->projections.resize(opt_flow_meas->observations.size());
    computeProjections(data->projections, last_state_t_ns);

    data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

    out_vis_queue->push(data);
  }

  last_processed_t_ns = last_state_t_ns;

  stats_sums_.add("measure", t_total.elapsed()).format("ms");

  return true;
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::logMargNullspace() {
  nullspace_marg_data.order = marg_data.order;
  if (config.vio_debug) {
    std::cout << "======== Marg nullspace ==========" << std::endl;
    stats_sums_.add("marg_ns", checkMargNullspace());
    std::cout << "=================================" << std::endl;
  } else {
    stats_sums_.add("marg_ns", checkMargNullspace());
  }
  stats_sums_.add("marg_ev", checkMargEigenvalues());
}

template <class Scalar_>
Eigen::VectorXd SqrtKeypointVoEstimator<Scalar_>::checkMargNullspace() const {
  return checkNullspace(nullspace_marg_data, frame_states, frame_poses,
                        config.vio_debug);
}

template <class Scalar_>
Eigen::VectorXd SqrtKeypointVoEstimator<Scalar_>::checkMargEigenvalues() const {
  return checkEigenvalues(nullspace_marg_data, false);
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::marginalize(
    const std::map<int64_t, int>& num_points_connected,
    const std::unordered_set<KeypointId>& lost_landmaks) {
  BASALT_ASSERT(frame_states.empty());

  Timer t_total;

  if (true) {
    // Marginalize

    // remove all frame_poses that are not kfs and not the current frame
    std::set<int64_t> non_kf_poses;
    for (const auto& kv : frame_poses) {
      if (kf_ids.count(kv.first) == 0 && kv.first != last_state_t_ns) {
        non_kf_poses.emplace(kv.first);
      }
    }

    for (int64_t id : non_kf_poses) {
      frame_poses.erase(id);
      lmdb.removeFrame(id);
      prev_opt_flow_res.erase(id);
    }

    auto kf_ids_all = kf_ids;
    std::set<FrameId> kfs_to_marg;
    while (kf_ids.size() > max_kfs) {
      int64_t id_to_marg = -1;

      // starting from the oldest kf (and skipping the newest 2 kfs), try to
      // find a kf that has less than a small percentage of it's landmarks
      // tracked by the current frame
      if (kf_ids.size() > 2) {
        // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
        auto end_minus_2 = std::prev(kf_ids.end(), 2);

        for (auto it = kf_ids.begin(); it != end_minus_2; ++it) {
          if (num_points_connected.count(*it) == 0 ||
              (num_points_connected.at(*it) / Scalar(num_points_kf.at(*it)) <
               Scalar(config.vio_kf_marg_feature_ratio))) {
            id_to_marg = *it;
            break;
          }
        }
      }

      // Note: This score function is taken from DSO, but it seems to mostly
      // marginalize the oldest keyframe. This may be due to the fact that
      // we don't have as long-lived landmarks, which may change if we ever
      // implement "rediscovering" of lost feature tracks by projecting
      // untracked landmarks into the localized frame.
      if (kf_ids.size() > 2 && id_to_marg < 0) {
        // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
        auto end_minus_2 = std::prev(kf_ids.end(), 2);

        int64_t last_kf = *kf_ids.crbegin();
        Scalar min_score = std::numeric_limits<Scalar>::max();
        int64_t min_score_id = -1;

        for (auto it1 = kf_ids.begin(); it1 != end_minus_2; ++it1) {
          // small distance to other keyframes --> higher score
          Scalar denom = 0;
          for (auto it2 = kf_ids.begin(); it2 != end_minus_2; ++it2) {
            denom += 1 / ((frame_poses.at(*it1).getPose().translation() -
                           frame_poses.at(*it2).getPose().translation())
                              .norm() +
                          Scalar(1e-5));
          }

          // small distance to latest kf --> lower score
          Scalar score =
              std::sqrt((frame_poses.at(*it1).getPose().translation() -
                         frame_poses.at(last_kf).getPose().translation())
                            .norm()) *
              denom;

          if (score < min_score) {
            min_score_id = *it1;
            min_score = score;
          }
        }

        id_to_marg = min_score_id;
      }

      // if no frame was selected, the logic above is faulty
      BASALT_ASSERT(id_to_marg >= 0);

      kfs_to_marg.emplace(id_to_marg);

      // Note: this looks like a leftover from VIO that is not doing anything in
      // VO -> we could check / compare / remove
      non_kf_poses.emplace(id_to_marg);

      kf_ids.erase(id_to_marg);
    }

    // Create AbsOrderMap entries that are in the marg prior or connected to the
    // keyframes that we marginalize
    // Create AbsOrderMap entries that are in the marg prior or connected to the
    // keyframes that we marginalize
    AbsOrderMap aom;
    {
      const auto& obs = lmdb.getObservations();

      aom.abs_order_map = marg_data.order.abs_order_map;
      aom.total_size = marg_data.order.total_size;
      aom.items = marg_data.order.items;

      for (const auto& kv : frame_poses) {
        if (aom.abs_order_map.count(kv.first) == 0) {
          bool add_pose = false;

          for (const auto& [host, target_map] : obs) {
            // if one of the host frames that we marg out
            if (kfs_to_marg.count(host.frame_id) > 0) {
              for (const auto& [target, obs_map] : target_map) {
                // has observations in the frame also add it to marg prior
                if (target.frame_id == kv.first) {
                  add_pose = true;
                  break;
                }
              }
            }
            // Break if we already found one observation.
            if (add_pose) break;
          }

          if (add_pose) {
            aom.abs_order_map[kv.first] =
                std::make_pair(aom.total_size, POSE_SIZE);

            aom.total_size += POSE_SIZE;
            aom.items++;
          }
        }
      }

      // If marg lost landmakrs add corresponding frames to linearization
      if (config.vio_marg_lost_landmarks) {
        for (const auto& lm_id : lost_landmaks) {
          const auto& lm = lmdb.getLandmark(lm_id);
          if (aom.abs_order_map.count(lm.host_kf_id.frame_id) == 0) {
            aom.abs_order_map[lm.host_kf_id.frame_id] =
                std::make_pair(aom.total_size, POSE_SIZE);

            aom.total_size += POSE_SIZE;
            aom.items++;
          }

          for (const auto& [target, o] : lm.obs) {
            if (aom.abs_order_map.count(target.frame_id) == 0) {
              aom.abs_order_map[target.frame_id] =
                  std::make_pair(aom.total_size, POSE_SIZE);

              aom.total_size += POSE_SIZE;
              aom.items++;
            }
          }
        }
      }
    }

    //    std::cout << "marg order" << std::endl;
    //    aom.print_order();

    //    std::cout << "marg prior order" << std::endl;
    //    marg_order.print_order();

    if (config.vio_debug) {
      std::cout << "non_kf_poses.size() " << non_kf_poses.size() << std::endl;
      for (const auto& v : non_kf_poses) std::cout << v << ' ';
      std::cout << std::endl;

      std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
      for (const auto& v : kfs_to_marg) std::cout << v << ' ';
      std::cout << std::endl;

      std::cout << "last_state_t_ns " << last_state_t_ns << std::endl;

      std::cout << "frame_poses.size() " << frame_poses.size() << std::endl;
      for (const auto& v : frame_poses) std::cout << v.first << ' ';
      std::cout << std::endl;
    }

    // Remove unconnected frames
    if (!kfs_to_marg.empty()) {
      for (auto it = kfs_to_marg.cbegin(); it != kfs_to_marg.cend();) {
        if (aom.abs_order_map.count(*it) == 0) {
          frame_poses.erase(*it);
          prev_opt_flow_res.erase(*it);
          lmdb.removeKeyframes({*it}, {}, {});
          it = kfs_to_marg.erase(it);
        } else {
          it++;
        }
      }
    }

    if (!kfs_to_marg.empty()) {
      Timer t_actual_marg;

      // Marginalize only if last state is a keyframe
      BASALT_ASSERT(kf_ids_all.count(last_state_t_ns) > 0);

      size_t asize = aom.total_size;
      //      double marg_prior_error;

      //      DenseAccumulator accum;
      //      accum.reset(asize);

      bool is_lin_sqrt = isLinearizationSqrt(config.vio_linearization_type);

      MatX Q2Jp_or_H;
      VecX Q2r_or_b;

      {
        // Linearize points
        Timer t_linearize;

        typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
        lqr_options.lb_options.huber_parameter = huber_thresh;
        lqr_options.lb_options.obs_std_dev = obs_std_dev;
        lqr_options.linearization_type = config.vio_linearization_type;

        auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
            this, aom, lqr_options, &marg_data, nullptr, &kfs_to_marg,
            &lost_landmaks);

        lqr->linearizeProblem();
        lqr->performQR();

        if (is_lin_sqrt && marg_data.is_sqrt) {
          lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
        } else {
          lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);
        }

        stats_sums_.add("marg_linearize", t_linearize.elapsed()).format("ms");
      }

      // Save marginalization prior
      if (out_marg_queue && !kfs_to_marg.empty()) {
        // int64_t kf_id = *kfs_to_marg.begin();

        {
          MargData::Ptr m(new MargData);
          m->aom = aom;

          if (is_lin_sqrt && marg_data.is_sqrt) {
            m->abs_H =
                (Q2Jp_or_H.transpose() * Q2Jp_or_H).template cast<double>();
            m->abs_b =
                (Q2Jp_or_H.transpose() * Q2r_or_b).template cast<double>();
          } else {
            m->abs_H = Q2Jp_or_H.template cast<double>();

            m->abs_b = Q2r_or_b.template cast<double>();
          }

          assign_cast_map_values(m->frame_poses, frame_poses);
          assign_cast_map_values(m->frame_states, frame_states);
          m->kfs_all = kf_ids_all;
          m->kfs_to_marg = kfs_to_marg;
          m->use_imu = false;

          for (int64_t t : m->kfs_all) {
            m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
          }

          out_marg_queue->push(m);
        }
      }

      std::set<int> idx_to_keep, idx_to_marg;
      for (const auto& kv : aom.abs_order_map) {
        if (kv.second.second == POSE_SIZE) {
          int start_idx = kv.second.first;
          if (kfs_to_marg.count(kv.first) == 0) {
            for (size_t i = 0; i < POSE_SIZE; i++)
              idx_to_keep.emplace(start_idx + i);
          } else {
            for (size_t i = 0; i < POSE_SIZE; i++)
              idx_to_marg.emplace(start_idx + i);
          }
        } else {
          BASALT_ASSERT(false);
        }
      }

      if (config.vio_debug) {
        std::cout << "keeping " << idx_to_keep.size() << " marg "
                  << idx_to_marg.size() << " total " << asize << std::endl;
        std::cout << " frame_poses " << frame_poses.size() << " frame_states "
                  << frame_states.size() << std::endl;
      }

      if (config.vio_debug || config.vio_extended_logging) {
        MatX Q2Jp_or_H_nullspace;
        VecX Q2r_or_b_nullspace;

        typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
        lqr_options.lb_options.huber_parameter = huber_thresh;
        lqr_options.lb_options.obs_std_dev = obs_std_dev;
        lqr_options.linearization_type = config.vio_linearization_type;

        nullspace_marg_data.order = marg_data.order;

        auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
            this, aom, lqr_options, &nullspace_marg_data, nullptr, &kfs_to_marg,
            &lost_landmaks);

        lqr->linearizeProblem();
        lqr->performQR();

        if (is_lin_sqrt && marg_data.is_sqrt) {
          lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
        } else {
          lqr->get_dense_H_b(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
        }

        MatX nullspace_sqrt_H_new;
        VecX nullspace_sqrt_b_new;

        if (is_lin_sqrt && marg_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
              Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
              nullspace_sqrt_H_new, nullspace_sqrt_b_new);
        } else if (marg_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqToSqrt(
              Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
              nullspace_sqrt_H_new, nullspace_sqrt_b_new);
        } else {
          MargHelper<Scalar>::marginalizeHelperSqToSq(
              Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
              nullspace_sqrt_H_new, nullspace_sqrt_b_new);
        }

        nullspace_marg_data.H = nullspace_sqrt_H_new;
        nullspace_marg_data.b = nullspace_sqrt_b_new;
      }

      MatX marg_sqrt_H_new;
      VecX marg_sqrt_b_new;

      {
        Timer t;
        if (is_lin_sqrt && marg_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
              Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_sqrt_H_new,
              marg_sqrt_b_new);
        } else if (marg_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqToSqrt(
              Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_sqrt_H_new,
              marg_sqrt_b_new);
        } else {
          MargHelper<Scalar>::marginalizeHelperSqToSq(
              Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_sqrt_H_new,
              marg_sqrt_b_new);
        }
        stats_sums_.add("marg_helper", t.elapsed()).format("ms");
      }

      for (auto& kv : frame_poses) {
        if (aom.abs_order_map.count(kv.first) > 0) {
          if (!kv.second.isLinearized()) kv.second.setLinTrue();
        }
      }

      for (const int64_t id : kfs_to_marg) {
        frame_poses.erase(id);
        prev_opt_flow_res.erase(id);
      }

      lmdb.removeKeyframes(kfs_to_marg, kfs_to_marg, kfs_to_marg);

      if (config.vio_marg_lost_landmarks) {
        for (const auto& lm_id : lost_landmaks) lmdb.removeLandmark(lm_id);
      }

      AbsOrderMap marg_order_new;

      for (const auto& kv : frame_poses) {
        if (aom.abs_order_map.count(kv.first) > 0) {
          marg_order_new.abs_order_map[kv.first] =
              std::make_pair(marg_order_new.total_size, POSE_SIZE);

          marg_order_new.total_size += POSE_SIZE;
          marg_order_new.items++;
        }
      }

      marg_data.H = marg_sqrt_H_new;
      marg_data.b = marg_sqrt_b_new;
      marg_data.order = marg_order_new;

      BASALT_ASSERT(size_t(marg_data.H.cols()) == marg_data.order.total_size);

      // Quadratic prior and "delta" of the current state to the original
      // linearization point give cost function
      //
      //    P(x) = 0.5 || J*(delta+x) + r ||^2.
      //
      // For marginalization this has been linearized at x=0 to give
      // linearization
      //
      //    P(x) = 0.5 || J*x + (J*delta + r) ||^2,
      //
      // with Jacobian J and residual J*delta + r.
      //
      // After marginalization, we recover the original form of the
      // prior. We are left with linearization (in sqrt form)
      //
      //    Pnew(x) = 0.5 || Jnew*x + res ||^2.
      //
      // To recover the original form with delta-independent r, we set
      //
      //    Pnew(x) = 0.5 || Jnew*(delta+x) + (res - Jnew*delta) ||^2,
      //
      // and thus rnew = (res - Jnew*delta).

      VecX delta;
      computeDelta(marg_data.order, delta);
      marg_data.b -= marg_data.H * delta;

      if (config.vio_debug || config.vio_extended_logging) {
        VecX delta;
        computeDelta(marg_data.order, delta);
        nullspace_marg_data.b -= nullspace_marg_data.H * delta;
      }

      stats_sums_.add("marg_total", t_actual_marg.elapsed()).format("ms");

      if (config.vio_debug) {
        std::cout << "marginalizaon done!!" << std::endl;
      }

      if (config.vio_debug || config.vio_extended_logging) {
        Timer t;
        logMargNullspace();
        stats_sums_.add("marg_log", t.elapsed()).format("ms");
      }
    }

    //    std::cout << "new marg prior order" << std::endl;
    //    marg_order.print_order();
  }

  stats_sums_.add("marginalize", t_total.elapsed()).format("ms");
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::optimize() {
  if (config.vio_debug) {
    std::cout << "=================================" << std::endl;
  }

  // harcoded configs
  //  bool scale_Jp = config.vio_scale_jacobian && is_qr_solver();
  //  bool scale_Jl = config.vio_scale_jacobian && is_qr_solver();

  // timing
  ExecutionStats stats;
  Timer timer_total;
  Timer timer_iteration;

  // construct order of states in linear system --> sort by ascending
  // timestamp
  AbsOrderMap aom;
  aom.abs_order_map = marg_data.order.abs_order_map;
  aom.total_size = marg_data.order.total_size;
  aom.items = marg_data.order.items;

  for (const auto& kv : frame_poses) {
    if (aom.abs_order_map.count(kv.first) == 0) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);
      aom.total_size += POSE_SIZE;
      aom.items++;
    }
  }

  // This is VO not VIO, so expect no IMU states
  BASALT_ASSERT(frame_states.empty());

  // TODO: Check why we get better accuracy with old SC loop. Possible culprits:
  // - different initial lambda (based on previous iteration)
  // - no landmark damping
  // - outlier removal after 4 iterations?
  lambda = Scalar(config.vio_lm_lambda_initial);

  // record stats
  stats.add("num_cams", frame_poses.size()).format("count");
  stats.add("num_lms", lmdb.numLandmarks()).format("count");
  stats.add("num_obs", lmdb.numObservations()).format("count");

  // setup landmark blocks
  typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
  lqr_options.lb_options.huber_parameter = huber_thresh;
  lqr_options.lb_options.obs_std_dev = obs_std_dev;
  lqr_options.linearization_type = config.vio_linearization_type;
  std::unique_ptr<LinearizationBase<Scalar, POSE_SIZE>> lqr;

  {
    Timer t;
    lqr = LinearizationBase<Scalar, POSE_SIZE>::create(this, aom, lqr_options,
                                                       &marg_data);
    stats.add("allocateLMB", t.reset()).format("ms");
    lqr->log_problem_stats(stats);
  }

  bool terminated = false;
  bool converged = false;
  std::string message;

  int it = 0;
  int it_rejected = 0;
  for (; it <= config.vio_max_iterations && !terminated;) {
    if (it > 0) {
      timer_iteration.reset();
    }

    Scalar error_total = 0;
    VecX Jp_column_norm2;

    // TODO: execution could be done staged

    Timer t;

    // linearize residuals
    bool numerically_valid;
    error_total = lqr->linearizeProblem(&numerically_valid);
    BASALT_ASSERT_STREAM(
        numerically_valid,
        "did not expect numerical failure during linearization");
    stats.add("linearizeProblem", t.reset()).format("ms");

    //      // compute pose jacobian norm squared for Jacobian scaling
    //      if (scale_Jp) {
    //        Jp_column_norm2 = lqr->getJp_diag2();
    //        stats.add("getJp_diag2", t.reset()).format("ms");
    //      }

    //      // scale landmark jacobians
    //      if (scale_Jl) {
    //        lqr->scaleJl_cols();
    //        stats.add("scaleJl_cols", t.reset()).format("ms");
    //      }

    // marginalize points in place
    lqr->performQR();
    stats.add("performQR", t.reset()).format("ms");

    if (config.vio_debug) {
      // TODO: num_points debug output missing
      std::cout << "[LINEARIZE] Error: " << error_total << " num points "
                << std::endl;
      std::cout << "Iteration " << it << " " << error_total << std::endl;
    }

    // compute pose jacobian scaling
    //    VecX jacobian_scaling;
    //    if (scale_Jp) {
    //      // TODO: what about jacobian scaling for SC solver?

    //      // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    //      // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    //      jacobian_scaling = (lqr_options.lb_options.jacobi_scaling_eps +
    //                          Jp_column_norm2.array().sqrt())
    //                             .inverse();
    //    }
    // if (config.vio_debug) {
    //   std::cout << "\t[INFO] Stage 1" << std::endl;
    //}

    // inner loop for backtracking in LM (still count as main iteration though)
    for (int j = 0; it <= config.vio_max_iterations && !terminated; j++) {
      if (j > 0) {
        timer_iteration.reset();
        if (config.vio_debug) {
          std::cout << "Iteration " << it << ", backtracking" << std::endl;
        }
      }

      {
        //        Timer t;

        // TODO: execution could be done staged

        // set (updated) damping for poses
        //        if (config.vio_lm_pose_damping_variant == 0) {
        //          lqr->setPoseDamping(lambda);
        //          stats.add("setPoseDamping", t.reset()).format("ms");
        //        }

        //        // scale landmark Jacobians only on the first inner iteration.
        //        if (scale_Jp && j == 0) {
        //          lqr->scaleJp_cols(jacobian_scaling);
        //          stats.add("scaleJp_cols", t.reset()).format("ms");
        //        }

        //        // set (updated) damping for landmarks
        //        if (config.vio_lm_landmark_damping_variant == 0) {
        //          lqr->setLandmarkDamping(lambda);
        //          stats.add("setLandmarkDamping", t.reset()).format("ms");
        //        }
      }

      // if (config.vio_debug) {
      //   std::cout << "\t[INFO] Stage 2 " << std::endl;
      // }

      VecX inc;
      {
        Timer t;

        // get dense reduced camera system
        MatX H;
        VecX b;

        lqr->get_dense_H_b(H, b);

        stats.add("get_dense_H_b", t.reset()).format("ms");

        int iter = 0;
        bool inc_valid = false;
        constexpr int max_num_iter = 3;

        while (iter < max_num_iter && !inc_valid) {
          VecX Hdiag_lambda = (H.diagonal() * lambda).cwiseMax(min_lambda);
          MatX H_copy = H;
          H_copy.diagonal() += Hdiag_lambda;

          Eigen::LDLT<Eigen::Ref<MatX>> ldlt(H_copy);
          inc = ldlt.solve(b);
          stats.add("solve", t.reset()).format("ms");

          if (!inc.array().isFinite().all()) {
            lambda = lambda_vee * lambda;
            lambda_vee *= vee_factor;
          } else {
            inc_valid = true;
          }
          iter++;
        }

        if (!inc_valid) {
          std::cerr << "Still invalid inc after " << max_num_iter
                    << " iterations." << std::endl;
        }
      }

      // backup state (then apply increment and check cost decrease)
      backup();

      // TODO: directly invert pose increment when solving; change SC
      // `updatePoints` to recieve unnegated increment

      // backsubstitute (with scaled pose increment)
      Scalar l_diff = 0;
      {
        // negate pose increment before point update
        inc = -inc;

        Timer t;
        l_diff = lqr->backSubstitute(inc);
        stats.add("backSubstitute", t.reset()).format("ms");
      }

      // undo jacobian scaling before applying increment to poses
      //      if (scale_Jp) {
      //        inc.array() *= jacobian_scaling.array();
      //      }

      // apply increment to poses
      for (auto& [frame_id, state] : frame_poses) {
        int idx = aom.abs_order_map.at(frame_id).first;
        state.applyInc(inc.template segment<POSE_SIZE>(idx));
      }

      // compute stepsize
      Scalar step_norminf = inc.array().abs().maxCoeff();

      // this is VO not VIO
      BASALT_ASSERT(frame_states.empty());

      // compute error update applying increment
      Scalar after_update_marg_prior_error = 0;
      Scalar after_update_vision_error = 0;

      {
        Timer t;
        computeError(after_update_vision_error);
        computeMargPriorError(marg_data, after_update_marg_prior_error);
        stats.add("computerError2", t.reset()).format("ms");
      }

      Scalar after_error_total =
          after_update_vision_error + after_update_marg_prior_error;

      // check cost decrease compared to quadratic model cost
      Scalar f_diff;
      bool step_is_valid = false;
      bool step_is_successful = false;
      Scalar relative_decrease = 0;
      {
        // compute actual cost decrease
        f_diff = error_total - after_error_total;

        relative_decrease = f_diff / l_diff;

        if (config.vio_debug) {
          std::cout << "\t[EVAL] error: {:.4e}, f_diff {:.4e} l_diff {:.4e} "
                       "step_quality {:.2e} step_size {:.2e}\n"_format(
                           after_error_total, f_diff, l_diff, relative_decrease,
                           step_norminf);
        }

        // TODO: consider to remove assert. For now we want to test if we even
        // run into the l_diff <= 0 case ever in practice
        // BASALT_ASSERT_STREAM(l_diff > 0, "l_diff " << l_diff);

        // l_diff <= 0 is a theoretical possibility if the model cost change is
        // tiny and becomes numerically negative (non-positive). It might not
        // occur since our linear systems are not that big (compared to large
        // scale BA for example) and we also abort optimization quite early and
        // usually don't have large damping (== tiny step size).
        step_is_valid = l_diff > 0;
        step_is_successful = step_is_valid && relative_decrease > 0;
      }

      double iteration_time = timer_iteration.elapsed();
      double cumulative_time = timer_total.elapsed();

      stats.add("iteration", iteration_time).format("ms");
      {
        basalt::MemoryInfo mi;
        if (get_memory_info(mi)) {
          stats.add("resident_memory", mi.resident_memory);
          stats.add("resident_memory_peak", mi.resident_memory_peak);
        }
      }

      if (step_is_successful) {
        BASALT_ASSERT(step_is_valid);

        if (config.vio_debug) {
          //          std::cout << "\t[ACCEPTED] lambda:" << lambda
          //                    << " Error: " << after_error_total << std::endl;

          std::cout << "\t[ACCEPTED] error: {:.4e}, lambda: {:.1e}, it_time: "
                       "{:.3f}s, total_time: {:.3f}s\n"
                       ""_format(after_error_total, lambda, iteration_time,
                                 cumulative_time);
        }

        lambda *= std::max<Scalar>(
            Scalar(1.0) / 3,
            1 - std::pow<Scalar>(2 * relative_decrease - 1, 3));
        lambda = std::max(min_lambda, lambda);

        lambda_vee = initial_vee;

        it++;

        // check function and parameter tolerance
        if ((f_diff > 0 && f_diff < Scalar(1e-6)) ||
            step_norminf < Scalar(1e-4)) {
          converged = true;
          terminated = true;
        }

        // stop inner lm loop
        break;
      } else {
        std::string reason = step_is_valid ? "REJECTED" : "INVALID";

        if (config.vio_debug) {
          //          std::cout << "\t[REJECTED] lambda:" << lambda
          //                    << " Error: " << after_error_total << std::endl;

          std::cout << "\t[{}] error: {}, lambda: {:.1e}, it_time:"
                       "{:.3f}s, total_time: {:.3f}s\n"
                       ""_format(reason, after_error_total, lambda,
                                 iteration_time, cumulative_time);
        }

        lambda = lambda_vee * lambda;
        lambda_vee *= vee_factor;

        //        lambda = std::max(min_lambda, lambda);
        //        lambda = std::min(max_lambda, lambda);

        restore();
        it++;
        it_rejected++;

        if (lambda > max_lambda) {
          terminated = true;
          message =
              "Solver did not converge and reached maximum damping lambda";
        }
      }
    }
  }

  stats.add("optimize", timer_total.elapsed()).format("ms");
  stats.add("num_it", it).format("count");
  stats.add("num_it_rejected", it_rejected).format("count");

  // TODO: call filterOutliers at least once (also for CG version)

  stats_all_.merge_all(stats);
  stats_sums_.merge_sums(stats);

  if (config.vio_debug) {
    if (!converged) {
      if (terminated) {
        std::cout << "Solver terminated early after {} iterations: {}"_format(
            it, message);
      } else {
        std::cout
            << "Solver did not converge after maximum number of {} iterations"_format(
                   it);
      }
    }

    stats.print();

    std::cout << "=================================" << std::endl;
  }
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::optimize_and_marg(
    const std::map<int64_t, int>& num_points_connected,
    const std::unordered_set<KeypointId>& lost_landmaks) {
  optimize();
  marginalize(num_points_connected, lost_landmaks);
}

template <class Scalar_>
void SqrtKeypointVoEstimator<Scalar_>::debug_finalize() {
  std::cout << "=== stats all ===\n";
  stats_all_.print();
  std::cout << "=== stats sums ===\n";
  stats_sums_.print();

  // save files
  stats_all_.save_json("stats_all.json");
  stats_sums_.save_json("stats_sums.json");
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class SqrtKeypointVoEstimator<double>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class SqrtKeypointVoEstimator<float>;
#endif
}  // namespace basalt
