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

#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

namespace basalt {

template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  FrameToFrameOpticalFlow(const VioConfig& config,
                          const basalt::Calibration<double>& calib)
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), config(config) {
    input_queue.set_capacity(10);

    this->calib = calib.cast<Scalar>();

    patch_coord = PatchT::pattern2.template cast<float>();

    if (calib.intrinsics.size() > 1) {
      Eigen::Matrix4d Ed;
      Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, Ed);
      E = Ed.cast<Scalar>();
    }

    processing_thread.reset(
        new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));
  }

  ~FrameToFrameOpticalFlow() { processing_thread->join(); }

  void processingLoop() {
    OpticalFlowInput::Ptr input_ptr;

    while (true) {
      input_queue.pop(input_ptr);

      if (!input_ptr.get()) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }

      processFrame(input_ptr->t_ns, input_ptr);
    }
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      transforms->observations.resize(calib.intrinsics.size());
      transforms->t_ns = t_ns;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(calib.intrinsics.size());

      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();

    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(calib.intrinsics.size());
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;

      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);
      }

      transforms = new_transforms;
      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();
    }

    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      output_queue->push(transforms);
    }

    frame_counter++;
  }

  void trackPoints(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                   const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_2) const {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto& kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f,
                                  std::hash<KeypointId>>
        result;

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);

        if (valid) {
          Eigen::AffineCompact2f transform_1_recovered = transform_2;

          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);

          if (valid) {
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();

            if (dist2 < config.optical_flow_max_recovered_dist2) {
              result[id] = transform_2;
            }
          }
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);

    tbb::parallel_for(range, compute_func);
    // compute_func(range);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t>& old_pyr,
                         const basalt::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid;
         level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      // Perform tracking on current level
      patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < config.optical_flow_max_iterations;
         iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      bool valid = dp.residual(img_2, transformed_pat, res);

      if (valid) {
        Vector3 inc = -dp.H_se2_inv_J_se2_T * res;
        transform *= SE2::exp(inc).matrix();

        const int filter_margin = 2;

        if (!img_2.InBounds(transform.translation(), filter_margin))
          patch_valid = false;
      } else {
        patch_valid = false;
      }
    }

    return patch_valid;
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto& kv : transforms->observations.at(0)) {
      pts0.emplace_back(kv.second.translation().cast<double>());
    }

    KeypointsData kd;

    detectKeypoints(pyramid->at(0).lvl(0), kd,
                    config.optical_flow_detection_grid_size, 1, pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;

    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      transforms->observations.at(0)[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    if (calib.intrinsics.size() > 1) {
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);

      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }
    }
  }

  void filterPoints() {
    if (calib.intrinsics.size() < 2) return;

    std::set<KeypointId> lm_to_remove;

    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto& kv : transforms->observations.at(1)) {
      auto it = transforms->observations.at(0).find(kv.first);

      if (it != transforms->observations.at(0).end()) {
        proj0.emplace_back(it->second.translation());
        proj1.emplace_back(kv.second.translation());
        kpid.emplace_back(kv.first);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[1].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    for (int id : lm_to_remove) {
      transforms->observations.at(1).erase(id);
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int64_t t_ns;

  size_t frame_counter;

  KeypointId last_keypoint_id;

  VioConfig config;
  basalt::Calibration<Scalar> calib;

  OpticalFlowResult::Ptr transforms;
  std::shared_ptr<std::vector<basalt::ManagedImagePyr<uint16_t>>> old_pyramid,
      pyramid;

  Matrix4 E;

  std::shared_ptr<std::thread> processing_thread;
};

}  // namespace basalt
