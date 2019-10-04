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

#include <memory>

#include <Eigen/Geometry>

#include <basalt/utils/vio_config.h>

#include <basalt/io/dataset_io.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>

#include <tbb/concurrent_queue.h>

namespace basalt {

using KeypointId = uint32_t;

struct OpticalFlowInput {
  using Ptr = std::shared_ptr<OpticalFlowInput>;

  int64_t t_ns;
  std::vector<ImageData> img_data;
};

struct OpticalFlowResult {
  using Ptr = std::shared_ptr<OpticalFlowResult>;

  int64_t t_ns;
  std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
      observations;

  OpticalFlowInput::Ptr input_images;
};

class OpticalFlowBase {
 public:
  using Ptr = std::shared_ptr<OpticalFlowBase>;

  tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr> input_queue;
  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* output_queue = nullptr;

  Eigen::MatrixXf patch_coord;
};

class OpticalFlowFactory {
 public:
  static OpticalFlowBase::Ptr getOpticalFlow(const VioConfig& config,
                                             const Calibration<double>& cam);
};
}  // namespace basalt
