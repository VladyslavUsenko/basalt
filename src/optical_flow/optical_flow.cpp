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

#include <basalt/optical_flow/optical_flow.h>

#include <basalt/optical_flow/frame_to_frame_optical_flow.h>
#include <basalt/optical_flow/patch_optical_flow.h>

namespace basalt {

OpticalFlowBase::Ptr OpticalFlowFactory::getOpticalFlow(
    const VioConfig& config, const Calibration<double>& cam) {
  OpticalFlowBase::Ptr res;

  if (config.optical_flow_type == "patch") {
    switch (config.optical_flow_pattern) {
      case 24:
        res.reset(new PatchOpticalFlow<float, Pattern24>(config, cam));
        break;

      case 52:
        res.reset(new PatchOpticalFlow<float, Pattern52>(config, cam));
        break;

      case 51:
        res.reset(new PatchOpticalFlow<float, Pattern51>(config, cam));
        break;

      case 50:
        res.reset(new PatchOpticalFlow<float, Pattern50>(config, cam));
        break;

      default:
        std::cerr << "config.optical_flow_pattern "
                  << config.optical_flow_pattern << " is not supported."
                  << std::endl;
        std::abort();
    }
  }

  if (config.optical_flow_type == "frame_to_frame") {
    switch (config.optical_flow_pattern) {
      case 24:
        res.reset(new FrameToFrameOpticalFlow<float, Pattern24>(config, cam));
        break;

      case 52:
        res.reset(new FrameToFrameOpticalFlow<float, Pattern52>(config, cam));
        break;

      case 51:
        res.reset(new FrameToFrameOpticalFlow<float, Pattern51>(config, cam));
        break;

      case 50:
        res.reset(new FrameToFrameOpticalFlow<float, Pattern50>(config, cam));
        break;

      default:
        std::cerr << "config.optical_flow_pattern "
                  << config.optical_flow_pattern << " is not supported."
                  << std::endl;
        std::abort();
    }
  }
  return res;
}
}  // namespace basalt
