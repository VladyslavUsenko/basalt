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

#include <basalt/utils/system_utils.h>

#include <fstream>

#if __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#elif __linux__
#include <unistd.h>

#include <sys/resource.h>
#endif

namespace basalt {

bool get_memory_info(MemoryInfo& info) {
#if __APPLE__
  mach_task_basic_info_data_t t_info;
  mach_msg_type_number_t t_info_count = MACH_TASK_BASIC_INFO_COUNT;

  if (KERN_SUCCESS != task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                (task_info_t)&t_info, &t_info_count)) {
    return false;
  }
  info.resident_memory = t_info.resident_size;
  info.resident_memory_peak = t_info.resident_size_max;

  /*
  struct rusage resource_usage;
  getrusage(RUSAGE_SELF, &resource_usage);
  info.resident_memory_peak = resource_usage.ru_maxrss;
  */

  return true;
#elif __linux__

  // get current memory first
  std::size_t program_size = 0;
  std::size_t resident_size = 0;

  std::ifstream fs("/proc/self/statm");
  if (fs.fail()) {
    return false;
  }
  fs >> program_size;
  fs >> resident_size;

  info.resident_memory = resident_size * sysconf(_SC_PAGESIZE);

  // get peak memory after that
  struct rusage resource_usage;
  getrusage(RUSAGE_SELF, &resource_usage);
  info.resident_memory_peak = resource_usage.ru_maxrss * 1024;

  return true;
#else
  return false;
#endif
}

}  // namespace basalt
