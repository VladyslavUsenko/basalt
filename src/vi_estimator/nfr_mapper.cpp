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

#include <basalt/optimization/accumulator.h>
#include <basalt/utils/keypoints.h>
#include <basalt/utils/nfr.h>
#include <basalt/utils/tracks.h>
#include <basalt/vi_estimator/nfr_mapper.h>

#include <basalt/hash_bow/hash_bow.h>

namespace basalt {

NfrMapper::NfrMapper(const Calibration<double>& calib, const VioConfig& config)
    : config(config),
      lambda(config.mapper_lm_lambda_min),
      min_lambda(config.mapper_lm_lambda_min),
      max_lambda(config.mapper_lm_lambda_max),
      lambda_vee(2) {
  this->calib = calib;
  this->obs_std_dev = config.mapper_obs_std_dev;
  this->huber_thresh = config.mapper_obs_huber_thresh;

  hash_bow_database.reset(new HashBow<256>(config.mapper_bow_num_bits));
}

void NfrMapper::addMargData(MargData::Ptr& data) {
  processMargData(*data);
  bool valid = extractNonlinearFactors(*data);

  if (valid) {
    for (const auto& kv : data->frame_poses) {
      PoseStateWithLin<double> p(kv.second.getT_ns(), kv.second.getPose());

      frame_poses[kv.first] = p;
    }

    for (const auto& kv : data->frame_states) {
      if (data->kfs_all.count(kv.first) > 0) {
        auto state = kv.second;
        PoseStateWithLin<double> p(state.getState().t_ns,
                                   state.getState().T_w_i);
        frame_poses[kv.first] = p;
      }
    }
  }
}

void NfrMapper::processMargData(MargData& m) {
  BASALT_ASSERT(m.aom.total_size == size_t(m.abs_H.cols()));

  //    std::cout << "rank " << m.abs_H.fullPivLu().rank() << " size "
  //              << m.abs_H.cols() << std::endl;

  AbsOrderMap aom_new;
  std::set<int> idx_to_keep;
  std::set<int> idx_to_marg;

  for (const auto& kv : m.aom.abs_order_map) {
    if (kv.second.second == POSE_SIZE) {
      for (size_t i = 0; i < POSE_SIZE; i++)
        idx_to_keep.emplace(kv.second.first + i);
      aom_new.abs_order_map.emplace(kv);
      aom_new.total_size += POSE_SIZE;
    } else if (kv.second.second == POSE_VEL_BIAS_SIZE) {
      if (m.kfs_all.count(kv.first) > 0) {
        for (size_t i = 0; i < POSE_SIZE; i++)
          idx_to_keep.emplace(kv.second.first + i);
        for (size_t i = POSE_SIZE; i < POSE_VEL_BIAS_SIZE; i++)
          idx_to_marg.emplace(kv.second.first + i);

        aom_new.abs_order_map[kv.first] =
            std::make_pair(aom_new.total_size, POSE_SIZE);
        aom_new.total_size += POSE_SIZE;

        PoseStateWithLin p = m.frame_states.at(kv.first);
        m.frame_poses[kv.first] = p;
        m.frame_states.erase(kv.first);
      } else {
        for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
          idx_to_marg.emplace(kv.second.first + i);
        m.frame_states.erase(kv.first);
      }
    } else {
      std::cerr << "Unknown size" << std::endl;
      std::abort();
    }

    //      std::cout << kv.first << " " << kv.second.first << " " <<
    //      kv.second.second
    //                << std::endl;
  }

  if (!idx_to_marg.empty()) {
    Eigen::MatrixXd marg_H_new;
    Eigen::VectorXd marg_b_new;
    BundleAdjustmentBase::marginalizeHelper(
        m.abs_H, m.abs_b, idx_to_keep, idx_to_marg, marg_H_new, marg_b_new);

    //    std::cout << "new rank " << marg_H_new.fullPivLu().rank() << " size "
    //              << marg_H_new.cols() << std::endl;

    m.abs_H = marg_H_new;
    m.abs_b = marg_b_new;
    m.aom = aom_new;
  }

  BASALT_ASSERT(m.aom.total_size == size_t(m.abs_H.cols()));

  // save image data
  {
    for (const auto& v : m.opt_flow_res) {
      img_data[v->t_ns] = v->input_images;
    }
  }
}

bool NfrMapper::extractNonlinearFactors(MargData& m) {
  size_t asize = m.aom.total_size;
  // std::cout << "asize " << asize << std::endl;

  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(m.abs_H);
  if (qr.rank() != m.abs_H.cols()) return false;

  Eigen::MatrixXd cov_old = qr.solve(Eigen::MatrixXd::Identity(asize, asize));

  int64_t kf_id = *m.kfs_to_marg.cbegin();
  int kf_start_idx = m.aom.abs_order_map.at(kf_id).first;

  auto state_kf = m.frame_poses.at(kf_id);

  Sophus::SE3d T_w_i_kf = state_kf.getPose();

  Eigen::Vector3d pos = T_w_i_kf.translation();
  Eigen::Vector3d yaw_dir_body =
      T_w_i_kf.so3().inverse() * Eigen::Vector3d::UnitX();

  Sophus::Matrix<double, 3, POSE_SIZE> d_pos_d_T_w_i;
  Sophus::Matrix<double, 1, POSE_SIZE> d_yaw_d_T_w_i;
  Sophus::Matrix<double, 2, POSE_SIZE> d_rp_d_T_w_i;

  absPositionError(T_w_i_kf, pos, &d_pos_d_T_w_i);
  yawError(T_w_i_kf, yaw_dir_body, &d_yaw_d_T_w_i);
  rollPitchError(T_w_i_kf, T_w_i_kf.so3(), &d_rp_d_T_w_i);

  {
    Eigen::MatrixXd J;
    J.setZero(POSE_SIZE, asize);
    J.block<3, POSE_SIZE>(0, kf_start_idx) = d_pos_d_T_w_i;
    J.block<1, POSE_SIZE>(3, kf_start_idx) = d_yaw_d_T_w_i;
    J.block<2, POSE_SIZE>(4, kf_start_idx) = d_rp_d_T_w_i;

    Sophus::Matrix6d cov_new = J * cov_old * J.transpose();

    // std::cout << "cov_new\n" << cov_new << std::endl;

    RollPitchFactor rpf;
    rpf.t_ns = kf_id;
    rpf.R_w_i_meas = T_w_i_kf.so3();

    if (!config.mapper_no_factor_weights) {
      rpf.cov_inv = cov_new.block<2, 2>(4, 4).inverse();
    } else {
      rpf.cov_inv.setIdentity();
    }

    if (m.use_imu) {
      roll_pitch_factors.emplace_back(rpf);
    }
  }

  for (int64_t other_id : m.kfs_all) {
    if (m.frame_poses.count(other_id) == 0 || other_id == kf_id) {
      continue;
    }

    auto state_o = m.frame_poses.at(other_id);

    Sophus::SE3d T_w_i_o = state_o.getPose();
    Sophus::SE3d T_kf_o = T_w_i_kf.inverse() * T_w_i_o;

    int o_start_idx = m.aom.abs_order_map.at(other_id).first;

    Sophus::Matrix6d d_res_d_T_w_i, d_res_d_T_w_j;
    relPoseError(T_kf_o, T_w_i_kf, T_w_i_o, &d_res_d_T_w_i, &d_res_d_T_w_j);

    Eigen::MatrixXd J;
    J.setZero(POSE_SIZE, asize);
    J.block<POSE_SIZE, POSE_SIZE>(0, kf_start_idx) = d_res_d_T_w_i;
    J.block<POSE_SIZE, POSE_SIZE>(0, o_start_idx) = d_res_d_T_w_j;

    Sophus::Matrix6d cov_new = J * cov_old * J.transpose();
    RelPoseFactor rpf;
    rpf.t_i_ns = kf_id;
    rpf.t_j_ns = other_id;
    rpf.T_i_j = T_kf_o;
    rpf.cov_inv.setIdentity();

    if (!config.mapper_no_factor_weights) {
      cov_new.ldlt().solveInPlace(rpf.cov_inv);
    }

    // std::cout << "rpf.cov_inv\n" << rpf.cov_inv << std::endl;

    rel_pose_factors.emplace_back(rpf);
  }

  return true;
}

void NfrMapper::optimize(int num_iterations) {
  AbsOrderMap aom;

  for (const auto& kv : frame_poses) {
    aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);
    aom.total_size += POSE_SIZE;
  }

  for (int iter = 0; iter < num_iterations; iter++) {
    auto t1 = std::chrono::high_resolution_clock::now();

    double rld_error;
    Eigen::aligned_vector<RelLinData> rld_vec;
    linearizeHelper(rld_vec, lmdb.getObservations(), rld_error);

    //      SparseHashAccumulator<double> accum;
    //      accum.reset(aom.total_size);

    //      for (auto& rld : rld_vec) {
    //        rld.invert_keypoint_hessians();

    //        Eigen::MatrixXd rel_H;
    //        Eigen::VectorXd rel_b;
    //        linearizeRel(rld, rel_H, rel_b);

    //        linearizeAbs(rel_H, rel_b, rld, aom, accum);
    //      }

    MapperLinearizeAbsReduce<SparseHashAccumulator<double>> lopt(aom,
                                                                 &frame_poses);
    tbb::blocked_range<Eigen::aligned_vector<RelLinData>::iterator> range(
        rld_vec.begin(), rld_vec.end());
    tbb::blocked_range<Eigen::aligned_vector<RollPitchFactor>::const_iterator>
        range1(roll_pitch_factors.begin(), roll_pitch_factors.end());
    tbb::blocked_range<Eigen::aligned_vector<RelPoseFactor>::const_iterator>
        range2(rel_pose_factors.begin(), rel_pose_factors.end());

    tbb::parallel_reduce(range, lopt);

    if (config.mapper_use_factors) {
      tbb::parallel_reduce(range1, lopt);
      tbb::parallel_reduce(range2, lopt);
    }

    double error_total = rld_error + lopt.rel_error + lopt.roll_pitch_error;

    std::cout << "[LINEARIZE] iter " << iter
              << " before_update_error: vision: " << rld_error
              << " rel_error: " << lopt.rel_error
              << " roll_pitch_error: " << lopt.roll_pitch_error
              << " total: " << error_total << std::endl;

    lopt.accum.iterative_solver = true;
    lopt.accum.print_info = true;

    lopt.accum.setup_solver();
    const Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

    bool converged = false;

    if (config.mapper_use_lm) {  // Use Levenberg–Marquardt
      bool step = false;
      int max_iter = 10;

      while (!step && max_iter > 0 && !converged) {
        Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
        for (int i = 0; i < Hdiag_lambda.size(); i++)
          Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

        const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);
        double max_inc = inc.array().abs().maxCoeff();
        if (max_inc < 1e-5) converged = true;

        backup();

        // apply increment to poses
        for (auto& kv : frame_poses) {
          int idx = aom.abs_order_map.at(kv.first).first;
          BASALT_ASSERT(!kv.second.isLinearized());
          kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
        }

        // Update points
        tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
        auto update_points_func = [&](const tbb::blocked_range<size_t>& r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            const auto& rld = rld_vec[i];
            updatePoints(aom, rld, inc);
          }
        };
        tbb::parallel_for(keys_range, update_points_func);

        double after_update_vision_error = 0;
        double after_rel_error = 0;
        double after_roll_pitch_error = 0;

        computeError(after_update_vision_error);
        if (config.mapper_use_factors) {
          computeRelPose(after_rel_error);
          computeRollPitch(after_roll_pitch_error);
        }

        double after_error_total = after_update_vision_error + after_rel_error +
                                   after_roll_pitch_error;

        double f_diff = (error_total - after_error_total);

        if (f_diff < 0) {
          std::cout << "\t[REJECTED] lambda:" << lambda << " f_diff: " << f_diff
                    << " max_inc: " << max_inc
                    << " vision_error: " << after_update_vision_error
                    << " rel_error: " << after_rel_error
                    << " roll_pitch_error: " << after_roll_pitch_error
                    << " total: " << after_error_total << std::endl;
          lambda = std::min(max_lambda, lambda_vee * lambda);
          lambda_vee *= 2;

          restore();
        } else {
          std::cout << "\t[ACCEPTED] lambda:" << lambda << " f_diff: " << f_diff
                    << " max_inc: " << max_inc
                    << " vision_error: " << after_update_vision_error
                    << " rel_error: " << after_rel_error
                    << " roll_pitch_error: " << after_roll_pitch_error
                    << " total: " << after_error_total << std::endl;

          lambda = std::max(min_lambda, lambda / 3);
          lambda_vee = 2;

          step = true;
        }

        max_iter--;

        if (after_error_total > error_total) {
          std::cout << "increased error after update!!!" << std::endl;
        }
      }
    } else {  // Use Gauss-Newton
      Eigen::VectorXd Hdiag_lambda = Hdiag * min_lambda;
      for (int i = 0; i < Hdiag_lambda.size(); i++)
        Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

      const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);
      double max_inc = inc.array().abs().maxCoeff();
      if (max_inc < 1e-5) converged = true;

      // apply increment to poses
      for (auto& kv : frame_poses) {
        int idx = aom.abs_order_map.at(kv.first).first;
        BASALT_ASSERT(!kv.second.isLinearized());
        kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
      }

      // Update points
      tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
      auto update_points_func = [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          const auto& rld = rld_vec[i];
          updatePoints(aom, rld, inc);
        }
      };
      tbb::parallel_for(keys_range, update_points_func);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    std::cout << "iter " << iter << " time : " << elapsed.count()
              << "(us),  num_states " << frame_states.size() << " num_poses "
              << frame_poses.size() << std::endl;

    if (converged) break;

    // std::cerr << "LT\n" << LT << std::endl;
    // std::cerr << "z_p\n" << z_p.transpose() << std::endl;
    // std::cerr << "inc\n" << inc.transpose() << std::endl;
  }
}

Eigen::aligned_map<int64_t, PoseStateWithLin<double>>&
NfrMapper::getFramePoses() {
  return frame_poses;
}

void NfrMapper::computeRelPose(double& rel_error) {
  rel_error = 0;

  for (const RelPoseFactor& rpf : rel_pose_factors) {
    const Sophus::SE3d& pose_i = frame_poses.at(rpf.t_i_ns).getPose();
    const Sophus::SE3d& pose_j = frame_poses.at(rpf.t_j_ns).getPose();

    Sophus::Vector6d res = relPoseError(rpf.T_i_j, pose_i, pose_j);

    rel_error += res.transpose() * rpf.cov_inv * res;
  }
}

void NfrMapper::computeRollPitch(double& roll_pitch_error) {
  roll_pitch_error = 0;

  for (const RollPitchFactor& rpf : roll_pitch_factors) {
    const Sophus::SE3d& pose = frame_poses.at(rpf.t_ns).getPose();

    Sophus::Vector2d res = rollPitchError(pose, rpf.R_w_i_meas);

    roll_pitch_error += res.transpose() * rpf.cov_inv * res;
  }
}

void NfrMapper::detect_keypoints() {
  std::vector<int64_t> keys;
  for (const auto& kv : img_data) {
    if (frame_poses.count(kv.first) > 0) {
      keys.emplace_back(kv.first);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, keys.size()),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t j = r.begin(); j != r.end(); ++j) {
          auto kv = img_data.find(keys[j]);
          if (kv->second.get()) {
            for (size_t i = 0; i < kv->second->img_data.size(); i++) {
              TimeCamId tcid(kv->first, i);
              KeypointsData& kd = feature_corners[tcid];

              if (!kv->second->img_data[i].img.get()) continue;

              const Image<const uint16_t> img =
                  kv->second->img_data[i].img->Reinterpret<const uint16_t>();

              detectKeypointsMapping(img, kd,
                                     config.mapper_detection_num_points);
              computeAngles(img, kd, true);
              computeDescriptors(img, kd);

              std::vector<bool> success;
              calib.intrinsics[tcid.cam_id].unproject(kd.corners, kd.corners_3d,
                                                      success);

              hash_bow_database->compute_bow(kd.corner_descriptors, kd.hashes,
                                             kd.bow_vector);

              hash_bow_database->add_to_database(tcid, kd.bow_vector);

              // std::cout << "bow " << kd.bow_vector.size() << " desc "
              //          << kd.corner_descriptors.size() << std::endl;
            }
          }
        }
      });

  auto t2 = std::chrono::high_resolution_clock::now();

  auto elapsed1 =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  std::cout << "Processed " << feature_corners.size() << " frames."
            << std::endl;

  std::cout << "Detection time: " << elapsed1.count() * 1e-6 << "s."
            << std::endl;
}

void NfrMapper::match_stereo() {
  // Pose of camera 1 (right) w.r.t camera 0 (left)
  const Sophus::SE3d T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];

  // Essential matrix
  Eigen::Matrix4d E;
  computeEssential(T_0_1, E);

  std::cout << "Matching " << img_data.size() << " stereo pairs..."
            << std::endl;

  int num_matches = 0;
  int num_inliers = 0;

  for (const auto& kv : img_data) {
    const TimeCamId tcid1(kv.first, 0), tcid2(kv.first, 1);

    MatchData md;
    md.T_i_j = T_0_1;

    const KeypointsData& kd1 = feature_corners[tcid1];
    const KeypointsData& kd2 = feature_corners[tcid2];

    matchDescriptors(kd1.corner_descriptors, kd2.corner_descriptors, md.matches,
                     config.mapper_max_hamming_distance,
                     config.mapper_second_best_test_ratio);

    num_matches += md.matches.size();

    findInliersEssential(kd1, kd2, E, 1e-3, md);

    if (md.inliers.size() > 16) {
      num_inliers += md.inliers.size();
      feature_matches[std::make_pair(tcid1, tcid2)] = md;
    }
  }

  std::cout << "Matched " << img_data.size() << " stereo pairs with "
            << num_inliers << " inlier matches (" << num_matches << " total)."
            << std::endl;
}

void NfrMapper::match_all() {
  std::vector<TimeCamId> keys;
  std::unordered_map<TimeCamId, size_t> id_to_key_idx;

  for (const auto& kv : feature_corners) {
    id_to_key_idx[kv.first] = keys.size();
    keys.push_back(kv.first);
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  struct match_pair {
    size_t i;
    size_t j;
    double score;
  };

  tbb::concurrent_vector<match_pair> ids_to_match;

  tbb::blocked_range<size_t> keys_range(0, keys.size());
  auto compute_pairs = [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const TimeCamId& tcid = keys[i];
      const KeypointsData& kd = feature_corners.at(tcid);

      std::vector<std::pair<TimeCamId, double>> results;

      hash_bow_database->querry_database(kd.bow_vector,
                                         config.mapper_num_frames_to_match,
                                         results, &tcid.frame_id);

      // std::cout << "Closest frames for " << tcid << ": ";
      for (const auto& otcid_score : results) {
        // std::cout << otcid_score.first << "(" << otcid_score.second << ")
        // ";
        if (otcid_score.first.frame_id != tcid.frame_id &&
            otcid_score.second > config.mapper_frames_to_match_threshold) {
          match_pair m;
          m.i = i;
          m.j = id_to_key_idx.at(otcid_score.first);
          m.score = otcid_score.second;

          ids_to_match.emplace_back(m);
        }
      }
      // std::cout << std::endl;
    }
  };

  tbb::parallel_for(keys_range, compute_pairs);
  // compute_pairs(keys_range);

  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "Matching " << ids_to_match.size() << " image pairs..."
            << std::endl;

  std::atomic<int> total_matched = 0;

  tbb::blocked_range<size_t> range(0, ids_to_match.size());
  auto match_func = [&](const tbb::blocked_range<size_t>& r) {
    int matched = 0;

    for (size_t j = r.begin(); j != r.end(); ++j) {
      const TimeCamId& id1 = keys[ids_to_match[j].i];
      const TimeCamId& id2 = keys[ids_to_match[j].j];

      const KeypointsData& f1 = feature_corners[id1];
      const KeypointsData& f2 = feature_corners[id2];

      MatchData md;

      matchDescriptors(f1.corner_descriptors, f2.corner_descriptors, md.matches,
                       70, 1.2);

      if (int(md.matches.size()) > config.mapper_min_matches) {
        matched++;

        findInliersRansac(f1, f2, config.mapper_ransac_threshold,
                          config.mapper_min_matches, md);
      }

      if (!md.inliers.empty()) feature_matches[std::make_pair(id1, id2)] = md;
    }
    total_matched += matched;
  };

  tbb::parallel_for(range, match_func);
  // match_func(range);

  auto t3 = std::chrono::high_resolution_clock::now();

  auto elapsed1 =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  auto elapsed2 =
      std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);

  //
  int num_matches = 0;
  int num_inliers = 0;

  for (const auto& kv : feature_matches) {
    num_matches += kv.second.matches.size();
    num_inliers += kv.second.inliers.size();
  }

  std::cout << "Matched " << ids_to_match.size() << " image pairs with "
            << num_inliers << " inlier matches (" << num_matches << " total)."
            << std::endl;

  std::cout << "DB query " << elapsed1.count() * 1e-6 << "s. matching "
            << elapsed2.count() * 1e-6
            << "s. Geometric verification attemts: " << total_matched << "."
            << std::endl;
}

void NfrMapper::build_tracks() {
  TrackBuilder trackBuilder;
  // Build: Efficient fusion of correspondences
  trackBuilder.Build(feature_matches);
  // Filter: Remove tracks that have conflict
  trackBuilder.Filter(config.mapper_min_track_length);
  // Export tree to usable data structure
  trackBuilder.Export(feature_tracks);

  // info
  size_t inlier_match_count = 0;
  for (const auto& it : feature_matches) {
    inlier_match_count += it.second.inliers.size();
  }

  size_t total_track_obs_count = 0;
  for (const auto& it : feature_tracks) {
    total_track_obs_count += it.second.size();
  }

  std::cout << "Built " << feature_tracks.size() << " feature tracks from "
            << inlier_match_count << " matches. Average track length is "
            << total_track_obs_count / (double)feature_tracks.size() << "."
            << std::endl;
}

void NfrMapper::setup_opt() {
  const double min_triang_distance2 = config.mapper_min_triangulation_dist *
                                      config.mapper_min_triangulation_dist;

  for (const auto& kv : feature_tracks) {
    if (kv.second.size() < 2) continue;

    // Take first observation as host
    auto it = kv.second.begin();
    TimeCamId tcid_h = it->first;

    FeatureId feat_id_h = it->second;
    Eigen::Vector2d pos_2d_h = feature_corners.at(tcid_h).corners[feat_id_h];
    Eigen::Vector4d pos_3d_h;
    calib.intrinsics[tcid_h.cam_id].unproject(pos_2d_h, pos_3d_h);

    it++;

    for (; it != kv.second.end(); it++) {
      TimeCamId tcid_o = it->first;

      FeatureId feat_id_o = it->second;
      Eigen::Vector2d pos_2d_o = feature_corners.at(tcid_o).corners[feat_id_o];
      Eigen::Vector4d pos_3d_o;
      calib.intrinsics[tcid_o.cam_id].unproject(pos_2d_o, pos_3d_o);

      Sophus::SE3d T_w_h = frame_poses.at(tcid_h.frame_id).getPose() *
                           calib.T_i_c[tcid_h.cam_id];
      Sophus::SE3d T_w_o = frame_poses.at(tcid_o.frame_id).getPose() *
                           calib.T_i_c[tcid_o.cam_id];

      Sophus::SE3d T_h_o = T_w_h.inverse() * T_w_o;

      if (T_h_o.translation().squaredNorm() < min_triang_distance2) continue;

      Eigen::Vector4d pos_3d =
          triangulate(pos_3d_h.head<3>(), pos_3d_o.head<3>(), T_h_o);

      if (!pos_3d.array().isFinite().all() || pos_3d[3] <= 0 || pos_3d[3] > 2.0)
        continue;

      KeypointPosition pos;
      pos.kf_id = tcid_h;
      pos.dir = StereographicParam<double>::project(pos_3d);
      pos.id = pos_3d[3];

      lmdb.addLandmark(kv.first, pos);

      for (const auto& obs_kv : kv.second) {
        KeypointObservation ko;
        ko.kpt_id = kv.first;
        ko.pos = feature_corners.at(obs_kv.first).corners[obs_kv.second];

        lmdb.addObservation(obs_kv.first, ko);
        // obs[tcid_h][obs_kv.first].emplace_back(ko);
      }
      break;
    }
  }
}

}  // namespace basalt
