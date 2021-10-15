/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2020-2021, Vladyslav Usenko and Nikolaus Demmel.
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

#include <chrono>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <Eigen/Dense>

namespace basalt {

template <class Clock = std::chrono::high_resolution_clock>
class Timer {
 public:
  /// start timer
  Timer() : start_(Clock::now()) {}

  /// return elapsed time in seconds
  double elapsed() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

  /// return elapsed time in seconds and reset timer
  double reset() {
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - start_).count();
    start_ = now;
    return elapsed;
  }

 private:
  std::chrono::time_point<Clock> start_;
};

template <class Scalar = double>
struct assign_op {
  void operator()(Scalar& lhs, const Scalar& rhs) { lhs = rhs; }
};

template <class Scalar = double>
struct plus_assign_op {
  void operator()(Scalar& lhs, const Scalar& rhs) { lhs += rhs; }
};

template <class T = Timer<>, class Assign = assign_op<>>
class ScopedTimer {
 public:
  explicit ScopedTimer(double& dest) : dest_(dest) {}
  ~ScopedTimer() { Assign()(dest_, timer_.elapsed()); }

 private:
  double& dest_;
  T timer_;
};

using ScopedTimerAdd = ScopedTimer<Timer<>, plus_assign_op<>>;

template <class F>
void log_timing(double& time, F fun) {
  Timer timer;
  fun();
  time = timer.elapsed();
}

template <class F>
void log_timing_add(double& time, F fun) {
  Timer timer;
  fun();
  time += timer.elapsed();
}

class ExecutionStats {
 public:
  struct Meta {
    inline Meta& format(const std::string& s) {
      format_ = s;
      return *this;
    }

    // overwrite the meta data from another object
    void set_meta(const Meta& other) { format_ = other.format_; }

    std::variant<std::vector<double>, std::vector<Eigen::VectorXd>> data_;

    std::string format_;
  };

  Meta& add(const std::string& name, double value);
  Meta& add(const std::string& name, const Eigen::VectorXd& value);
  Meta& add(const std::string& name, const Eigen::VectorXf& value);

  void merge_all(const ExecutionStats& other);

  void merge_sums(const ExecutionStats& other);

  void print() const;

  bool save_json(const std::string& path) const;

 private:
  std::unordered_map<std::string, Meta> stats_;
  std::vector<std::string> order_;
};

}  // namespace basalt
