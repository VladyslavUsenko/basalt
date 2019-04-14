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
#include <thread>

#include <basalt/utils/imu_types.h>

namespace basalt {

class MargDataSaver {
 public:
  using Ptr = std::shared_ptr<MargDataSaver>;

  MargDataSaver(const std::string& path);
  ~MargDataSaver() {
    saving_thread->join();
    saving_img_thread->join();
  }
  tbb::concurrent_bounded_queue<MargData::Ptr> in_marg_queue;

 private:
  std::shared_ptr<std::thread> saving_thread;
  std::shared_ptr<std::thread> saving_img_thread;

  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr> save_image_queue;
};

class MargDataLoader {
 public:
  using Ptr = std::shared_ptr<MargDataLoader>;

  MargDataLoader();

  void start(const std::string& path);
  ~MargDataLoader() { processing_thread->join(); }

  tbb::concurrent_bounded_queue<MargData::Ptr>* out_marg_queue;

 private:
  std::shared_ptr<std::thread> processing_thread;
};
}  // namespace basalt
