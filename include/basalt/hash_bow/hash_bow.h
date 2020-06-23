#pragma once

#include <array>
#include <bitset>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <basalt/utils/common_types.h>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

namespace basalt {

template <size_t N>
class HashBow {
 public:
  HashBow(size_t num_bits) : num_bits(num_bits < 32 ? num_bits : 32) {
    static_assert(N < 512,
                  "This implementation of HashBow only supports the descriptor "
                  "length below 512.");
  }

  inline FeatureHash compute_hash(const std::bitset<N>& descriptor) const {
    FeatureHash res;
    for (size_t i = 0; i < num_bits; ++i) {
      res[i] = descriptor[word_bit_permutation[i]];
    }
    return res;
  }

  inline void compute_bow(const std::vector<std::bitset<N>>& descriptors,
                          std::vector<FeatureHash>& hashes,
                          HashBowVector& bow_vector) const {
    size_t descriptors_size = descriptors.size();
    hashes.resize(descriptors_size);

    std::unordered_map<FeatureHash, double> bow_map;
    bow_map.clear();
    bow_map.reserve(descriptors_size);

    for (size_t i = 0; i < descriptors_size; i++) {
      hashes[i] = compute_hash(descriptors[i]);
      bow_map[hashes[i]] += 1.0;
    }

    bow_vector.clear();

    double l1_sum = 0;
    for (const auto& kv : bow_map) {
      bow_vector.emplace_back(kv);
      l1_sum += std::abs(kv.second);
    }

    for (auto& kv : bow_vector) {
      kv.second /= l1_sum;
    }
  }

  inline void add_to_database(const TimeCamId& tcid,
                              const HashBowVector& bow_vector) {
    for (const auto& kv : bow_vector) {
      // std::pair<TimeCamId, double> p = std::make_pair(tcid, kv.second);
      inverted_index[kv.first].emplace_back(tcid, kv.second);
    }
  }

  inline void querry_database(
      const HashBowVector& bow_vector, size_t num_results,
      std::vector<std::pair<TimeCamId, double>>& results,
      const int64_t* max_t_ns = nullptr) const {
    results.clear();

    std::unordered_map<TimeCamId, double> scores;

    for (const auto& kv : bow_vector) {
      const auto range_it = inverted_index.find(kv.first);

      if (range_it != inverted_index.end())
        for (const auto& v : range_it->second) {
          // if there is a maximum query time select only the frames that have
          // timestamp below max_t_ns
          if (!max_t_ns || v.first.frame_id < (*max_t_ns))
            scores[v.first] += std::abs(kv.second - v.second) -
                               std::abs(kv.second) - std::abs(v.second);
        }
    }

    results.reserve(scores.size());

    for (const auto& kv : scores)
      results.emplace_back(kv.first, -kv.second / 2.0);

    if (results.size() > num_results) {
      std::partial_sort(
          results.begin(), results.begin() + num_results, results.end(),
          [](const auto& a, const auto& b) { return a.second > b.second; });

      results.resize(num_results);
    }
  }

 protected:
  constexpr static const size_t random_bit_permutation[512] = {
      484, 458, 288, 170, 215, 424, 41,  38,  293, 96,  172, 428, 508, 52,  370,
      1,   182, 472, 89,  339, 273, 234, 98,  217, 73,  195, 307, 306, 113, 429,
      161, 443, 364, 439, 301, 247, 325, 24,  490, 366, 75,  7,   464, 232, 49,
      196, 144, 69,  470, 387, 3,   86,  361, 313, 396, 356, 94,  201, 291, 360,
      107, 251, 413, 393, 296, 124, 308, 146, 298, 160, 121, 302, 151, 345, 336,
      26,  63,  238, 79,  267, 262, 437, 433, 350, 53,  134, 194, 452, 114, 54,
      82,  214, 191, 242, 482, 37,  432, 311, 130, 460, 422, 221, 271, 192, 474,
      46,  289, 34,  20,  95,  463, 499, 159, 272, 481, 129, 448, 173, 323, 258,
      416, 229, 334, 510, 461, 263, 362, 346, 39,  500, 381, 401, 492, 299, 33,
      169, 241, 11,  254, 449, 199, 486, 400, 365, 70,  436, 108, 19,  233, 505,
      152, 6,   480, 468, 278, 426, 253, 471, 328, 327, 139, 29,  27,  488, 332,
      290, 412, 164, 259, 352, 222, 186, 32,  319, 410, 211, 405, 187, 213, 507,
      205, 395, 62,  178, 36,  140, 87,  491, 351, 450, 314, 77,  342, 132, 133,
      477, 103, 389, 206, 197, 324, 485, 425, 297, 231, 123, 447, 126, 9,   64,
      181, 40,  14,  5,   261, 431, 333, 223, 4,   138, 220, 76,  44,  300, 331,
      78,  193, 497, 403, 435, 275, 147, 66,  368, 141, 451, 225, 250, 61,  18,
      444, 208, 380, 109, 255, 337, 372, 212, 359, 457, 31,  398, 354, 219, 117,
      248, 392, 203, 88,  479, 509, 149, 120, 145, 51,  15,  367, 190, 163, 417,
      454, 329, 183, 390, 83,  404, 249, 81,  264, 445, 317, 179, 244, 473, 71,
      111, 118, 209, 171, 224, 459, 446, 104, 13,  377, 200, 414, 198, 420, 226,
      153, 384, 25,  441, 305, 338, 316, 483, 184, 402, 48,  131, 502, 252, 469,
      12,  167, 243, 373, 35,  127, 341, 455, 379, 210, 340, 128, 430, 57,  434,
      330, 415, 494, 142, 355, 282, 322, 65,  105, 421, 68,  409, 466, 245, 59,
      269, 112, 386, 257, 256, 93,  174, 16,  60,  143, 343, 115, 506, 276, 10,
      496, 489, 235, 47,  136, 22,  165, 204, 42,  465, 440, 498, 312, 504, 116,
      419, 185, 303, 218, 353, 283, 374, 2,   177, 137, 240, 102, 309, 292, 85,
      453, 388, 397, 438, 281, 279, 442, 110, 55,  101, 100, 150, 375, 406, 157,
      23,  0,   237, 376, 236, 216, 8,   154, 91,  456, 423, 176, 427, 284, 30,
      84,  349, 335, 56,  270, 227, 286, 168, 239, 122, 478, 162, 475, 166, 17,
      348, 285, 175, 155, 266, 382, 304, 268, 180, 295, 125, 371, 467, 277, 294,
      58,  347, 72,  280, 50,  287, 511, 80,  260, 326, 495, 45,  106, 399, 369,
      503, 357, 315, 418, 487, 99,  43,  320, 188, 407, 246, 501, 119, 158, 274,
      408, 230, 358, 90,  148, 363, 207, 344, 265, 462, 189, 310, 385, 67,  28,
      383, 378, 156, 394, 97,  476, 493, 321, 411, 228, 21,  391, 202, 92,  318,
      74,  135};

  constexpr static std::array<size_t, FEATURE_HASH_MAX_SIZE>
  compute_permutation() {
    std::array<size_t, FEATURE_HASH_MAX_SIZE> res{};
    size_t j = 0;
    for (size_t i = 0; i < 512 && j < FEATURE_HASH_MAX_SIZE; ++i) {
      if (random_bit_permutation[i] < N) {
        res[j] = random_bit_permutation[i];
        j++;
      }
    }

    return res;
  }

  constexpr static const std::array<size_t, FEATURE_HASH_MAX_SIZE>
      word_bit_permutation = compute_permutation();

  size_t num_bits;

  tbb::concurrent_unordered_map<
      FeatureHash, tbb::concurrent_vector<std::pair<TimeCamId, double>>,
      std::hash<FeatureHash>>
      inverted_index;
};

}  // namespace basalt
