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

#include <basalt/io/marg_data_io.h>

#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/filesystem.h>

namespace basalt {

MargDataSaver::MargDataSaver(const std::string& path) {
  fs::remove_all(path);
  fs::create_directory(path);

  save_image_queue.set_capacity(300);

  std::string img_path = path + "/images/";
  fs::create_directory(img_path);

  in_marg_queue.set_capacity(1000);

  auto save_func = [&, path]() {
    basalt::MargData::Ptr data;

    std::unordered_set<int64_t> processed_opt_flow;

    while (true) {
      in_marg_queue.pop(data);

      if (data.get()) {
        int64_t kf_id = *data->kfs_to_marg.begin();

        std::string p = path + "/" + std::to_string(kf_id) + ".cereal";
        std::ofstream os(p, std::ios::binary);

        {
          cereal::BinaryOutputArchive archive(os);
          archive(*data);
        }
        os.close();

        for (const auto& d : data->opt_flow_res) {
          if (processed_opt_flow.count(d->t_ns) == 0) {
            save_image_queue.push(d);
            processed_opt_flow.emplace(d->t_ns);
          }
        }

      } else {
        save_image_queue.push(nullptr);
        break;
      }
    }

    std::cout << "Finished MargDataSaver" << std::endl;
  };

  auto save_image_func = [&, img_path]() {
    basalt::OpticalFlowResult::Ptr data;

    while (true) {
      save_image_queue.pop(data);

      if (data.get()) {
        std::string p = img_path + "/" + std::to_string(data->t_ns) + ".cereal";
        std::ofstream os(p, std::ios::binary);

        {
          cereal::BinaryOutputArchive archive(os);
          archive(data);
        }
        os.close();
      } else {
        break;
      }
    }

    std::cout << "Finished image MargDataSaver" << std::endl;
  };

  saving_thread.reset(new std::thread(save_func));
  saving_img_thread.reset(new std::thread(save_image_func));
}  // namespace basalt

MargDataLoader::MargDataLoader() : out_marg_queue(nullptr) {}

void MargDataLoader::start(const std::string& path) {
  if (!fs::exists(path))
    std::cerr << "No marg. data found in " << path << std::endl;

  auto func = [&, path]() {
    std::string img_path = path + "/images/";

    std::unordered_set<uint64_t> saved_images;

    std::map<int64_t, OpticalFlowResult::Ptr> opt_flow_res;

    for (const auto& entry : fs::directory_iterator(img_path)) {
      OpticalFlowResult::Ptr data;
      // std::cout << entry.path() << std::endl;
      std::ifstream is(entry.path(), std::ios::binary);
      {
        cereal::BinaryInputArchive archive(is);
        archive(data);
      }
      is.close();
      opt_flow_res[data->t_ns] = data;
    }

    std::map<int64_t, std::string> filenames;

    for (auto& p : fs::directory_iterator(path)) {
      std::string filename = p.path().filename();
      if (!std::isdigit(filename[0])) continue;

      size_t lastindex = filename.find_last_of(".");
      std::string rawname = filename.substr(0, lastindex);

      int64_t t_ns = std::stol(rawname);

      filenames.emplace(t_ns, filename);
    }

    for (const auto& kv : filenames) {
      basalt::MargData::Ptr data(new basalt::MargData);

      std::string p = path + "/" + kv.second;
      std::ifstream is(p, std::ios::binary);

      {
        cereal::BinaryInputArchive archive(is);
        archive(*data);
      }
      is.close();

      for (const auto& d : data->kfs_all) {
        data->opt_flow_res.emplace_back(opt_flow_res.at(d));
      }

      out_marg_queue->push(data);
    }

    out_marg_queue->push(nullptr);

    std::cout << "Finished MargDataLoader" << std::endl;
  };

  processing_thread.reset(new std::thread(func));
}
}  // namespace basalt

namespace cereal {

template <class Archive, class T>
void save(Archive& ar, const basalt::ManagedImage<T>& m) {
  ar(m.w);
  ar(m.h);
  ar(cereal::binary_data(m.ptr, sizeof(T) * m.w * m.h));
}

template <class Archive, class T>
void load(Archive& ar, basalt::ManagedImage<T>& m) {
  size_t w;
  size_t h;
  ar(w);
  ar(h);
  m.Reinitialise(w, h);
  ar(cereal::binary_data(m.ptr, sizeof(T) * m.w * m.h));
}

template <class Archive>
void serialize(Archive& ar, basalt::OpticalFlowResult& m) {
  ar(m.t_ns);
  ar(m.observations);
  ar(m.input_images);
}

template <class Archive>
void serialize(Archive& ar, basalt::OpticalFlowInput& m) {
  ar(m.t_ns);
  ar(m.img_data);
}

template <class Archive>
void serialize(Archive& ar, basalt::ImageData& m) {
  ar(m.exposure);
  ar(m.img);
}

template <class Archive>
static void serialize(Archive& ar, Eigen::AffineCompact2f& m) {
  ar(m.matrix());
}
}  // namespace cereal
