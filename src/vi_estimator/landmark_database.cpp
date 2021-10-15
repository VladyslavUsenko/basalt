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

#include <algorithm>

#include <basalt/vi_estimator/landmark_database.h>

namespace basalt {

template <class Scalar_>
void LandmarkDatabase<Scalar_>::addLandmark(KeypointId lm_id,
                                            const Keypoint<Scalar> &pos) {
  auto &kpt = kpts[lm_id];
  kpt.direction = pos.direction;
  kpt.inv_dist = pos.inv_dist;
  kpt.host_kf_id = pos.host_kf_id;
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeFrame(const FrameId &frame) {
  for (auto it = kpts.begin(); it != kpts.end();) {
    for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
      if (it2->first.frame_id == frame)
        it2 = removeLandmarkObservationHelper(it, it2);
      else
        it2++;
    }

    if (it->second.obs.size() < min_num_obs) {
      it = removeLandmarkHelper(it);
    } else {
      ++it;
    }
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeKeyframes(
    const std::set<FrameId> &kfs_to_marg,
    const std::set<FrameId> &poses_to_marg,
    const std::set<FrameId> &states_to_marg_all) {
  for (auto it = kpts.begin(); it != kpts.end();) {
    if (kfs_to_marg.count(it->second.host_kf_id.frame_id) > 0) {
      it = removeLandmarkHelper(it);
    } else {
      for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
        FrameId fid = it2->first.frame_id;
        if (poses_to_marg.count(fid) > 0 || states_to_marg_all.count(fid) > 0 ||
            kfs_to_marg.count(fid) > 0)
          it2 = removeLandmarkObservationHelper(it, it2);
        else
          it2++;
      }

      if (it->second.obs.size() < min_num_obs) {
        it = removeLandmarkHelper(it);
      } else {
        ++it;
      }
    }
  }
}

template <class Scalar_>
std::vector<TimeCamId> LandmarkDatabase<Scalar_>::getHostKfs() const {
  std::vector<TimeCamId> res;

  for (const auto &kv : observations) res.emplace_back(kv.first);

  return res;
}

template <class Scalar_>
std::vector<const Keypoint<Scalar_> *>
LandmarkDatabase<Scalar_>::getLandmarksForHost(const TimeCamId &tcid) const {
  std::vector<const Keypoint<Scalar> *> res;

  for (const auto &[k, obs] : observations.at(tcid))
    for (const auto &v : obs) res.emplace_back(&kpts.at(v));

  return res;
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::addObservation(
    const TimeCamId &tcid_target, const KeypointObservation<Scalar> &o) {
  auto it = kpts.find(o.kpt_id);
  BASALT_ASSERT(it != kpts.end());

  it->second.obs[tcid_target] = o.pos;

  observations[it->second.host_kf_id][tcid_target].insert(it->first);
}

template <class Scalar_>
Keypoint<Scalar_> &LandmarkDatabase<Scalar_>::getLandmark(KeypointId lm_id) {
  return kpts.at(lm_id);
}

template <class Scalar_>
const Keypoint<Scalar_> &LandmarkDatabase<Scalar_>::getLandmark(
    KeypointId lm_id) const {
  return kpts.at(lm_id);
}

template <class Scalar_>
const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>
    &LandmarkDatabase<Scalar_>::getObservations() const {
  return observations;
}

template <class Scalar_>
const Eigen::aligned_unordered_map<KeypointId, Keypoint<Scalar_>>
    &LandmarkDatabase<Scalar_>::getLandmarks() const {
  return kpts;
}

template <class Scalar_>
bool LandmarkDatabase<Scalar_>::landmarkExists(int lm_id) const {
  return kpts.count(lm_id) > 0;
}

template <class Scalar_>
size_t LandmarkDatabase<Scalar_>::numLandmarks() const {
  return kpts.size();
}

template <class Scalar_>
int LandmarkDatabase<Scalar_>::numObservations() const {
  int num_observations = 0;

  for (const auto &[_, val_map] : observations) {
    for (const auto &[_, val] : val_map) {
      num_observations += val.size();
    }
  }

  return num_observations;
}

template <class Scalar_>
int LandmarkDatabase<Scalar_>::numObservations(KeypointId lm_id) const {
  return kpts.at(lm_id).obs.size();
}

template <class Scalar_>
typename LandmarkDatabase<Scalar_>::MapIter
LandmarkDatabase<Scalar_>::removeLandmarkHelper(
    LandmarkDatabase<Scalar>::MapIter it) {
  auto host_it = observations.find(it->second.host_kf_id);

  for (const auto &[k, v] : it->second.obs) {
    auto target_it = host_it->second.find(k);
    target_it->second.erase(it->first);

    if (target_it->second.empty()) host_it->second.erase(target_it);
  }

  if (host_it->second.empty()) observations.erase(host_it);

  return kpts.erase(it);
}

template <class Scalar_>
typename Keypoint<Scalar_>::MapIter
LandmarkDatabase<Scalar_>::removeLandmarkObservationHelper(
    LandmarkDatabase<Scalar>::MapIter it,
    typename Keypoint<Scalar>::MapIter it2) {
  auto host_it = observations.find(it->second.host_kf_id);
  auto target_it = host_it->second.find(it2->first);
  target_it->second.erase(it->first);

  if (target_it->second.empty()) host_it->second.erase(target_it);
  if (host_it->second.empty()) observations.erase(host_it);

  return it->second.obs.erase(it2);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeLandmark(KeypointId lm_id) {
  auto it = kpts.find(lm_id);
  if (it != kpts.end()) removeLandmarkHelper(it);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeObservations(
    KeypointId lm_id, const std::set<TimeCamId> &obs) {
  auto it = kpts.find(lm_id);
  BASALT_ASSERT(it != kpts.end());

  for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
    if (obs.count(it2->first) > 0) {
      it2 = removeLandmarkObservationHelper(it, it2);
    } else
      it2++;
  }

  if (it->second.obs.size() < min_num_obs) {
    removeLandmarkHelper(it);
  }
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

// Note: double specialization is unconditional, b/c NfrMapper depends on it.
//#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class LandmarkDatabase<double>;
//#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class LandmarkDatabase<float>;
#endif

}  // namespace basalt
