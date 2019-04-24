// Adapted from OpenMVG

// Copyright (c) 2012, 2013 Pierre MOULON
//               2018 Nikolaus DEMMEL

// This file was originally part of OpenMVG, an Open Multiple View Geometry C++
// library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Implementation of [1] an efficient algorithm to compute track from pairwise
//  correspondences.
//
//  [1] Pierre Moulon and Pascal Monasse,
//    "Unordered feature tracking made fast and easy" CVMP 2012.
//
// It tracks the position of features along the series of image from pairwise
//  correspondences.
//
// From map<[imageI,ImageJ], [indexed matches array] > it builds tracks.
//
// Usage :
//  //---------------------------------------
//  // Compute tracks from matches
//  //---------------------------------------
//  TrackBuilder trackBuilder;
//  FeatureTracks tracks;
//  trackBuilder.Build(matches); // Build: Efficient fusion of correspondences
//  trackBuilder.Filter();       // Filter: Remove tracks that have conflict
//  trackBuilder.Export(tracks); // Export tree to usable data structure

#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <basalt/utils/common_types.h>
#include <basalt/utils/union_find.h>

namespace basalt {

/// TrackBuild class creates feature tracks from matches
struct TrackBuilder {
  std::map<ImageFeaturePair, TrackId> map_node_to_index;
  UnionFind uf_tree;

  /// Build tracks for a given series of pairWise matches
  void Build(const Matches& map_pair_wise_matches) {
    // 1. We need to know how much single set we will have.
    //   i.e each set is made of a tuple : (imageIndex, featureIndex)
    std::set<ImageFeaturePair> allFeatures;
    // For each couple of images list the used features
    for (const auto& iter : map_pair_wise_matches) {
      const auto I = iter.first.first;
      const auto J = iter.first.second;
      const MatchData& matchData = iter.second;

      // Retrieve all shared features and add them to a set
      for (const auto& match : matchData.inliers) {
        allFeatures.emplace(I, match.first);
        allFeatures.emplace(J, match.second);
      }
    }

    // 2. Build the 'flat' representation where a tuple (the node)
    //  is attached to a unique index.
    TrackId cpt = 0;
    for (const auto& feat : allFeatures) {
      map_node_to_index.emplace(feat, cpt);
      ++cpt;
    }
    // Clean some memory
    allFeatures.clear();

    // 3. Add the node and the pairwise correpondences in the UF tree.
    uf_tree.InitSets(map_node_to_index.size());

    // 4. Union of the matched features corresponding UF tree sets
    for (const auto& iter : map_pair_wise_matches) {
      const auto I = iter.first.first;
      const auto J = iter.first.second;
      const MatchData& matchData = iter.second;
      for (const auto& match : matchData.inliers) {
        const ImageFeaturePair pairI(I, match.first);
        const ImageFeaturePair pairJ(J, match.second);
        // Link feature correspondences to the corresponding containing sets.
        uf_tree.Union(map_node_to_index[pairI], map_node_to_index[pairJ]);
      }
    }
  }

  /// Remove bad tracks (too short or track with ids collision)
  bool Filter(size_t minimumTrackLength = 2) {
    // Remove bad tracks:
    // - track that are too short,
    // - track with id conflicts:
    //    i.e. tracks that have many times the same image index

    // From the UF tree, create tracks of the image indexes.
    //  If an image index appears twice the track must disappear
    //  If a track is too short it has to be removed.
    std::map<TrackId, std::set<TimeCamId>> tracks;

    std::set<TrackId> problematic_track_id;
    // Build tracks from the UF tree, track problematic ids.
    for (const auto& iter : map_node_to_index) {
      const TrackId track_id = uf_tree.Find(iter.second);
      if (problematic_track_id.count(track_id) != 0) {
        continue;  // Track already marked
      }

      const ImageFeaturePair& feat = iter.first;

      if (tracks[track_id].count(feat.first)) {
        problematic_track_id.insert(track_id);
      } else {
        tracks[track_id].insert(feat.first);
      }
    }

    // - track that are too short,
    for (const auto& val : tracks) {
      if (val.second.size() < minimumTrackLength) {
        problematic_track_id.insert(val.first);
      }
    }

    for (uint32_t& root_index : uf_tree.m_cc_parent) {
      if (problematic_track_id.count(root_index) > 0) {
        // reset selected root
        uf_tree.m_cc_size[root_index] = 1;
        root_index = UnionFind::InvalidIndex();
      }
    }
    return false;
  }

  /// Return the number of connected set in the UnionFind structure (tree
  /// forest)
  size_t TrackCount() const {
    std::set<TrackId> parent_id(uf_tree.m_cc_parent.begin(),
                                uf_tree.m_cc_parent.end());
    // Erase the "special marker" that depicted rejected tracks
    parent_id.erase(UnionFind::InvalidIndex());
    return parent_id.size();
  }

  /// Export tracks as a map (each entry is a map of imageId and
  /// featureIndex):
  ///  {TrackIndex => {imageIndex => featureIndex}}
  void Export(FeatureTracks& tracks) {
    tracks.clear();
    for (const auto& iter : map_node_to_index) {
      const TrackId track_id = uf_tree.Find(iter.second);
      const ImageFeaturePair& feat = iter.first;
      // ensure never add rejected elements (track marked as invalid)
      if (track_id != UnionFind::InvalidIndex()) {
        tracks[track_id].emplace(feat);
      }
    }
  }
};

/// Find common tracks between images.
bool GetTracksInImages(const std::set<TimeCamId>& image_ids,
                       const FeatureTracks& all_tracks,
                       std::vector<TrackId>& shared_track_ids) {
  shared_track_ids.clear();

  // Go along the tracks
  for (const auto& kv_track : all_tracks) {
    // Look if the track contains the provided view index & save the point ids
    size_t observed_image_count = 0;
    for (const auto& imageId : image_ids) {
      if (kv_track.second.count(imageId) > 0) {
        ++observed_image_count;
      } else {
        break;
      }
    }

    if (observed_image_count == image_ids.size()) {
      shared_track_ids.push_back(kv_track.first);
    }
  }
  return !shared_track_ids.empty();
}

/// Find all tracks in an image.
bool GetTracksInImage(const TimeCamId& image_id,
                      const FeatureTracks& all_tracks,
                      std::vector<TrackId>& track_ids) {
  std::set<TimeCamId> image_set;
  image_set.insert(image_id);
  return GetTracksInImages(image_set, all_tracks, track_ids);
}

/// Find shared tracks between map and image
bool GetSharedTracks(const TimeCamId& image_id, const FeatureTracks& all_tracks,
                     const Landmarks& landmarks,
                     std::vector<TrackId>& track_ids) {
  track_ids.clear();
  for (const auto& kv : landmarks) {
    const TrackId trackId = kv.first;
    if (all_tracks.at(trackId).count(image_id) > 0) {
      track_ids.push_back(trackId);
    }
  }
  return !track_ids.empty();
}

}  // namespace basalt
