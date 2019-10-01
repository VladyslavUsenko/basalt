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

#include <basalt/io/dataset_io.h>
#include <basalt/io/dataset_io_euroc.h>
#include <basalt/io/dataset_io_kitti.h>
#include <basalt/io/dataset_io_rosbag.h>
#include <basalt/io/dataset_io_uzh.h>

namespace basalt {

DatasetIoInterfacePtr DatasetIoFactory::getDatasetIo(
    const std::string &dataset_type, bool load_mocap_as_gt) {
  if (dataset_type == "euroc") {
    // return DatasetIoInterfacePtr();
    return DatasetIoInterfacePtr(new EurocIO(load_mocap_as_gt));
  } else if (dataset_type == "bag") {
    return DatasetIoInterfacePtr(new RosbagIO);
  } else if (dataset_type == "uzh") {
    return DatasetIoInterfacePtr(new UzhIO);
  } else if (dataset_type == "kitti") {
    return DatasetIoInterfacePtr(new KittiIO);
  } else {
    std::cerr << "Dataset type " << dataset_type << " is not supported"
              << std::endl;
    std::abort();
  }
}

}  // namespace basalt
