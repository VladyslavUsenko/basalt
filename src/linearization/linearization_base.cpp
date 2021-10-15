/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2021, Vladyslav Usenko and Nikolaus Demmel.
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

#include <basalt/linearization/linearization_abs_qr.hpp>
#include <basalt/linearization/linearization_abs_sc.hpp>
#include <basalt/linearization/linearization_base.hpp>
#include <basalt/linearization/linearization_rel_sc.hpp>

#include <magic_enum.hpp>

namespace basalt {

bool isLinearizationSqrt(const LinearizationType& type) {
  switch (type) {
    case LinearizationType::ABS_QR:
      return true;
    case LinearizationType::ABS_SC:
    case LinearizationType::REL_SC:
      return false;
    default:
      BASALT_ASSERT_STREAM(false, "Linearization type is not supported.");
      return false;
  }
}

template <typename Scalar_, int POSE_SIZE_>
std::unique_ptr<LinearizationBase<Scalar_, POSE_SIZE_>>
LinearizationBase<Scalar_, POSE_SIZE_>::create(
    BundleAdjustmentBase<Scalar>* estimator, const AbsOrderMap& aom,
    const Options& options, const MargLinData<Scalar>* marg_lin_data,
    const ImuLinData<Scalar>* imu_lin_data,
    const std::set<FrameId>* used_frames,
    const std::unordered_set<KeypointId>* lost_landmarks,
    int64_t last_state_to_marg) {
  //  std::cout << "Creaing Linearization of type "
  //            << magic_enum::enum_name(options.linearization_type) <<
  //            std::endl;

  switch (options.linearization_type) {
    case LinearizationType::ABS_QR:
      return std::make_unique<LinearizationAbsQR<Scalar, POSE_SIZE>>(
          estimator, aom, options, marg_lin_data, imu_lin_data, used_frames,
          lost_landmarks, last_state_to_marg);

    case LinearizationType::ABS_SC:
      return std::make_unique<LinearizationAbsSC<Scalar, POSE_SIZE>>(
          estimator, aom, options, marg_lin_data, imu_lin_data, used_frames,
          lost_landmarks, last_state_to_marg);

    case LinearizationType::REL_SC:
      return std::make_unique<LinearizationRelSC<Scalar, POSE_SIZE>>(
          estimator, aom, options, marg_lin_data, imu_lin_data, used_frames,
          lost_landmarks, last_state_to_marg);

    default:
      std::cerr << "Could not select a valid linearization." << std::endl;
      std::abort();
  }
}

// //////////////////////////////////////////////////////////////////
// instatiate factory templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
// Scalar=double, POSE_SIZE=6
template std::unique_ptr<LinearizationBase<double, 6>>
LinearizationBase<double, 6>::create(
    BundleAdjustmentBase<double>* estimator, const AbsOrderMap& aom,
    const Options& options, const MargLinData<double>* marg_lin_data,
    const ImuLinData<double>* imu_lin_data,
    const std::set<FrameId>* used_frames,
    const std::unordered_set<KeypointId>* lost_landmarks,
    int64_t last_state_to_marg);
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
// Scalar=float, POSE_SIZE=6
template std::unique_ptr<LinearizationBase<float, 6>>
LinearizationBase<float, 6>::create(
    BundleAdjustmentBase<float>* estimator, const AbsOrderMap& aom,
    const Options& options, const MargLinData<float>* marg_lin_data,
    const ImuLinData<float>* imu_lin_data, const std::set<FrameId>* used_frames,
    const std::unordered_set<KeypointId>* lost_landmarks,
    int64_t last_state_to_marg);
#endif

}  // namespace basalt
