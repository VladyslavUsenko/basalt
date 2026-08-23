#include "basalt/vi_estimator/local_mapper.h"

#include <basalt/hash_bow/hash_bow.h>
#include <basalt/utils/keypoints.h>

#include <algorithm>
#include <basalt/camera/stereographic_param.hpp>
#include <chrono>
#include <iostream>

namespace basalt {

// ═══════════════════════════════════════════════════════════════════
// Construction / Destruction / Lifecycle
// ═══════════════════════════════════════════════════════════════════

LocalMapper::LocalMapper(const Calibration<double>& calib,
                         const VioConfig& config)
    : NfrMapper(calib, config) {
    // Replace NfrMapper's TBB-backed HashBow with the STL-backed variant that
    // supports RemoveKeyframes (needed for keyframe culling).
    hash_bow_database =
        std::make_shared<HashBowStl<256>>(config.mapper_bow_num_bits);
}

LocalMapper::~LocalMapper() { Stop(); }

void LocalMapper::Stop() {
    std::cout << "inside stop function" << std::endl;
    mpStopLocalMapping = true;
    if (mpLocalMappingThread.joinable()) {
        std::cout << "attempting to stop" << std::endl;
        mpLocalMappingThread.join();
        std::cout << "able to stop local mapping thread" << std::endl;
    }
    mpTrackBuilder.ResetTrackId();
    std::cout << "track builder ID reset and mapper stopped" << std::endl;
}

void LocalMapper::Initialise() {
    mpStopLocalMapping = false;
    mpIsTrackBuilderInitialised = false;
    mpRetiredTrackIds.clear();
    mpNewKeyframesForTracking.clear();
    mpLatestKeyframesMatches.clear();

    // Spawn the mapping thread. It busy-waits in MapLocally until the queue
    // is wired (mpIsMargDataInputQueueSet == true).
    mpLocalMappingThread = std::thread(&LocalMapper::MapLocally, this);
}

void LocalMapper::SetMarginalisationDataInputQueue(
    tbb::concurrent_bounded_queue<MargData::Ptr>* queue) {
    mpMargInputQueue = queue;
    mpIsMargDataInputQueueSet = true;
}

void LocalMapper::SetVIOPoseUpdateCallback(PoseUpdateCallback cb) {
    mpVioPoseUpdateCallback = std::move(cb);
}

// ═══════════════════════════════════════════════════════════════════
// MapLocally — thread entry point
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::MapLocally() {
    // Wait until the queue is wired.
    while (!mpStopLocalMapping && !mpIsMargDataInputQueueSet) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    while (!mpStopLocalMapping) {
        auto t1 = std::chrono::high_resolution_clock::now();

        // Block on the first item so the thread sleeps (zero CPU) when idle,
        // then try_pop any further items that accumulated during the last
        // cycle.
        std::vector<MargData::Ptr> vecData;
        bool nullReceived = false;
        MargData::Ptr data;

        mpMargInputQueue->pop(data);  // blocking — sleeps until VIO pushes
        if (!data) {
            std::cout << "[Local Mapper] got shutdown sentinel" << std::endl;
            break;
        }
        vecData.push_back(data);

        while (mpMargInputQueue->try_pop(data)) {
            if (!data) {
                nullReceived = true;
                break;
            }
            vecData.push_back(data);
        }
        auto tStep = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(tStep - t1);
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] Queue wait time taken: "
                      << elapsed.count() * 1e-6 << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] Procesing " << vecData.size()
                      << " marginalisation data packets" << std::endl;
        mpNewKeyframesForTracking.clear();
        mpLatestKeyframesMatches.clear();
        for (MargData::Ptr& packet : vecData) {
            IngestMargData(packet);
        }
        vecData.clear();

        for (auto it = img_data.begin(); it != img_data.end();) {
            if (mpNewKeyframesForTracking.count(it->first) == 0) {
                it = img_data.erase(it);
            } else
                ++it;
        }
        auto tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] IngestMargData time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        if (mpNewKeyframesForTracking.empty()) continue;

        tStep = std::chrono::high_resolution_clock::now();
        detect_keypoints();  // inherited — only touches img_data (now filtered)
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] detect_keypoints time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        if (calib.T_i_c.size() > 1) {
            tStep = std::chrono::high_resolution_clock::now();
            match_stereo();  // inherited — writes to feature_matches
            tEnd = std::chrono::high_resolution_clock::now();
            if (mpVioDebugMode)
                std::cout
                    << "[Local Mapper] match_stereo time taken: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                           tEnd - tStep)
                               .count() *
                           1e-6
                    << "s" << std::endl;

            tStep = std::chrono::high_resolution_clock::now();
        }
        MatchLocal();  // BoW cross-frame matching for new KFs
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] MatchLocal time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        CollectNewKeyframesAfterMatching();  // promote to
                                             // mpLatestKeyframesMatches
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] CollectNewKeyframesAfterMatching time "
                         "taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        build_tracks();  // LocalMapper (shadowing NfrMapper)
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] build_tracks time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        setup_opt();  // LocalMapper (shadowing NfrMapper)
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] setup_opt time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        CullRedundantKeyframes();
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] CullRedundantKeyframes time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        optimize(mpOptIterations);
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] optimize (1st pass) time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        filterOutliers(mpFilterOutlierThreshold, 4);
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] filterOutliers time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        optimize(mpOptIterations);
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] optimize (2nd pass) time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        // Publish an immutable local-map snapshot for the GUI. This is the only
        // point in the cycle where frame_poses and lmdb are quiescent on the
        // mapping thread, so the copy is internally consistent and race-free.
        tStep = std::chrono::high_resolution_clock::now();
        if (out_vis_queue) {
            LocalMapperVisualizationData::Ptr vis_data =
                std::make_shared<LocalMapperVisualizationData>();
            vis_data->t_ns =
                frame_poses.empty() ? 0 : frame_poses.rbegin()->first;
            get_current_points(vis_data->points, vis_data->point_ids);
            vis_data->keyframes.reserve(frame_poses.size());
            for (const auto& kv : frame_poses)
                vis_data->keyframes.emplace_back(kv.second.getPose());
            out_vis_queue->try_push(std::move(vis_data));  // never block mapper
        }
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] Visualization publish time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        tStep = std::chrono::high_resolution_clock::now();
        if (mpVioPoseUpdateCallback) mpVioPoseUpdateCallback(frame_poses);
        tEnd = std::chrono::high_resolution_clock::now();
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] VIO pose update callback time taken: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             tEnd - tStep)
                                 .count() *
                             1e-6
                      << "s" << std::endl;

        auto elapsed1 =
            std::chrono::duration_cast<std::chrono::microseconds>(tEnd - t1);
        if (mpVioDebugMode) {
            std::cout << "[Local Mapper] Total Local Mapping Time: "
                      << elapsed1.count() * 1e-6 << "s" << std::endl;
            std::cout << "[Local Mapper] "
                         "====================================================="
                         "======="
                      << std::endl;
        }

        // If the sentinel arrived mid-drain (alongside real data), we still
        // processed all data above — now shut down cleanly.
        if (nullReceived) break;
    }
}

// ═══════════════════════════════════════════════════════════════════
// IngestMargData
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::IngestMargData(MargData::Ptr& data) {
    // Step 1 — detect new KFs (those in kfs_all but not yet in frame_poses).
    for (const int64_t id : data->kfs_all) {
        if (frame_poses.count(id) == 0) {
            mpNewKeyframesForTracking.emplace(id);
        }
    }

    // Step 2 — inherited factor extraction (mutates data).
    processMargData(*data);
    bool valid = extractNonlinearFactors(*data);

    // Step 3 — update frame_poses.
    // New KFs are added unconditionally; existing KFs updated only if valid.
    // After processMargData, POSE_VEL_BIAS entries have been moved from
    // data->frame_states into data->frame_poses.
    for (const auto& kv : data->frame_poses) {
        if (mpNewKeyframesForTracking.count(kv.first)) {
            frame_poses[kv.first] = PoseStateWithLin<double>(
                kv.second.getT_ns(), kv.second.getPose());
        }
    }

    if (valid) {
        // Update existing KF poses from the latest VIO marginalisation.
        for (const auto& kv : data->frame_poses) {
            if (mpNewKeyframesForTracking.count(kv.first) == 0) {
                PoseStateWithLin<double> p(kv.second.getT_ns(),
                                           kv.second.getPose());
                frame_poses[kv.first] = p;
            }
        }

        for (const auto& kv : data->frame_states) {
            if (mpNewKeyframesForTracking.count(kv.first) == 0) {
                if (data->kfs_all.count(kv.first) > 0) {
                    auto state = kv.second;
                    PoseStateWithLin<double> p(state.getState().t_ns,
                                               state.getState().T_w_i);
                    frame_poses[kv.first] = p;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// CollectNewKeyframesAfterMatching
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::CollectNewKeyframesAfterMatching() {
    for (const auto& kv : feature_matches) {
        const auto& pair = kv.first;
        if (mpNewKeyframesForTracking.count(pair.first.frame_id) ||
            mpNewKeyframesForTracking.count(pair.second.frame_id)) {
            mpLatestKeyframesMatches[pair] = kv.second;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MatchLocal — BoW cross-frame matching restricted to new KFs
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::MatchLocal() {
    std::vector<TimeCamId> keys;
    std::unordered_map<TimeCamId, size_t> id_to_key_idx;

    for (const auto& kv : feature_corners) {
        id_to_key_idx[kv.first] = keys.size();
        keys.push_back(kv.first);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    struct match_pair {
        size_t i;
        size_t j;
        double score;
    };

    tbb::concurrent_vector<match_pair> ids_to_match;

    tbb::blocked_range<size_t> keys_range(0, keys.size());
    auto compute_pairs = [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            const TimeCamId& tcid = keys[i];
            // Only query from new KFs — this is the key difference from
            // match_all.
            if (mpNewKeyframesForTracking.count(tcid.frame_id) > 0) {
                const KeypointsData& kd = feature_corners.at(tcid);

                std::vector<std::pair<TimeCamId, double>> results;

                hash_bow_database->querry_database(
                    kd.bow_vector, config.mapper_num_frames_to_match, results,
                    &tcid.frame_id);

                for (const auto& otcid_score : results) {
                    if (otcid_score.first.frame_id != tcid.frame_id &&
                        otcid_score.second >
                            config.mapper_frames_to_match_threshold) {
                        match_pair m;
                        m.i = i;
                        m.j = id_to_key_idx.at(otcid_score.first);
                        m.score = otcid_score.second;
                        ids_to_match.emplace_back(m);
                    }
                }
            }
        }
    };

    tbb::parallel_for(keys_range, compute_pairs);

    auto t2 = std::chrono::high_resolution_clock::now();

    if (mpVioDebugMode)
        std::cout << "[Local Mapper] Matching " << ids_to_match.size()
                  << " image pairs..." << std::endl;

    std::atomic<int> total_matched{0};

    tbb::blocked_range<size_t> range(0, ids_to_match.size());
    auto match_func = [&](const tbb::blocked_range<size_t>& r) {
        int matched = 0;

        for (size_t j = r.begin(); j != r.end(); ++j) {
            const TimeCamId& id1 = keys[ids_to_match[j].i];
            const TimeCamId& id2 = keys[ids_to_match[j].j];

            const KeypointsData& f1 = feature_corners[id1];
            const KeypointsData& f2 = feature_corners[id2];

            MatchData md;

            matchDescriptors(f1.corner_descriptors, f2.corner_descriptors,
                             md.matches, 70, 1.2);

            if (static_cast<int>(md.matches.size()) >
                config.mapper_min_matches) {
                matched++;
                findInliersRansac(f1, f2, config.mapper_ransac_threshold,
                                  config.mapper_min_matches, md);
            }

            if (!md.inliers.empty()) {
                feature_matches[std::make_pair(id1, id2)] =
                    std::allocate_shared<MatchData>(
                        Eigen::aligned_allocator<MatchData>{}, md);
            }
        }
        total_matched += matched;
    };

    tbb::parallel_for(range, match_func);

    auto t3 = std::chrono::high_resolution_clock::now();

    auto elapsed1 =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto elapsed2 =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);

    int num_matches = 0;
    int num_inliers = 0;
    for (const auto& kv : mpLatestKeyframesMatches) {
        num_matches += kv.second->matches.size();
        num_inliers += kv.second->inliers.size();
    }

    if (mpVioDebugMode) {
        std::cout << "[Local Mapper] Matched " << ids_to_match.size()
                  << " image pairs with " << num_inliers << " inlier matches ("
                  << num_matches << " total)." << std::endl;

        std::cout << "DB query " << elapsed1.count() * 1e-6 << "s. matching "
                  << elapsed2.count() * 1e-6
                  << "s. Geometric verification attempts: " << total_matched
                  << "." << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════
// build_tracks — incremental track building (shadows NfrMapper)
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::build_tracks() {
    // Convert mpLatestKeyframesMatches (std::unordered_map) to Matches (TBB
    // map) because TrackBuilder::Build uses the Matches API.
    Matches tbb_matches;
    for (const auto& kv : mpLatestKeyframesMatches)
        tbb_matches.insert({kv.first, kv.second});

    if (!mpIsTrackBuilderInitialised) {
        mpTrackBuilder.Build(tbb_matches);
        mpTrackBuilder.Filter(config.mapper_min_track_length);
        feature_tracks.clear();
        mpTrackBuilder.Export(feature_tracks);
        mpIsTrackBuilderInitialised = true;
    } else {
        feature_tracks.clear();
        mpTrackBuilder.AddNewMatches(tbb_matches, lmdb, feature_tracks,
                                     mpRetiredTrackIds);
    }
}

// ═══════════════════════════════════════════════════════════════════
// setup_opt — landmark DB update (shadows NfrMapper)
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::setup_opt() {
    const double min_triang_dist2 = config.mapper_min_triangulation_dist *
                                    config.mapper_min_triangulation_dist;

    // Step A — retire merged landmarks.
    for (const TrackId retired : mpRetiredTrackIds) {
        if (lmdb.landmarkExists(retired)) {
            lmdb.removeLandmark(retired);
        }
    }
    mpRetiredTrackIds.clear();

    // Step B — iterate updated feature tracks.
    for (const auto& kv : feature_tracks) {
        if (kv.second.size() < 2) continue;

        // Add landmark iff not yet known to lmdb — skip-if-exists preserves
        // optimised geometry from previous iterations.
        if (!lmdb.landmarkExists(kv.first)) {
            auto it_h = kv.second.begin();
            TimeCamId tcid_h = it_h->first;
            FeatureId fid_h = it_h->second;

            Eigen::Vector2d pos_2d_h =
                feature_corners.at(tcid_h).corners[fid_h];
            Eigen::Vector4d pos_3d_h;
            calib.intrinsics[tcid_h.cam_id].unproject(pos_2d_h, pos_3d_h);

            bool triangulated = false;
            for (auto it_o = std::next(it_h); it_o != kv.second.end(); ++it_o) {
                TimeCamId tcid_o = it_o->first;
                FeatureId fid_o = it_o->second;
                if (feature_corners.count(tcid_o) == 0) continue;
                if (!frame_poses.count(tcid_h.frame_id)) continue;
                if (!frame_poses.count(tcid_o.frame_id)) continue;

                Eigen::Vector2d pos_2d_o =
                    feature_corners.at(tcid_o).corners[fid_o];
                Eigen::Vector4d pos_3d_o;
                calib.intrinsics[tcid_o.cam_id].unproject(pos_2d_o, pos_3d_o);

                Sophus::SE3d T_w_h = frame_poses.at(tcid_h.frame_id).getPose() *
                                     calib.T_i_c[tcid_h.cam_id];
                Sophus::SE3d T_w_o = frame_poses.at(tcid_o.frame_id).getPose() *
                                     calib.T_i_c[tcid_o.cam_id];
                Sophus::SE3d T_h_o = T_w_h.inverse() * T_w_o;

                if (T_h_o.translation().squaredNorm() < min_triang_dist2)
                    continue;

                Eigen::Vector4d pos_3d =
                    triangulate(pos_3d_h.head<3>(), pos_3d_o.head<3>(), T_h_o);
                if (!pos_3d.array().isFinite().all() || pos_3d[3] <= 0 ||
                    pos_3d[3] > 2.0)
                    continue;

                Keypoint<Scalar> kpt;
                kpt.host_kf_id = tcid_h;
                kpt.direction = StereographicParam<double>::project(pos_3d);
                kpt.inv_dist = pos_3d[3];
                lmdb.addLandmark(kv.first, kpt);
                triangulated = true;
                break;
            }
            if (!triangulated) continue;
        }

        // Add observations (idempotent — safe to repeat).
        for (const auto& obs_kv : kv.second) {
            if (!frame_poses.count(obs_kv.first.frame_id)) continue;
            if (feature_corners.count(obs_kv.first) == 0) continue;
            KeypointObservation<Scalar> ko;
            ko.kpt_id = kv.first;
            ko.pos = feature_corners.at(obs_kv.first).corners[obs_kv.second];
            lmdb.addObservation(obs_kv.first, ko);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// ComputeCovisibility
// ═══════════════════════════════════════════════════════════════════

size_t LocalMapper::ComputeCovisibility(int64_t tid_a, int64_t tid_b,
                                        int num_cameras) {
    size_t covisibility = 0;
    for (int i = 0; i < num_cameras; i++) {
        for (int j = 0; j < num_cameras; j++) {
            covisibility += lmdb.getObservationsCountForPair(
                TimeCamId(tid_a, i), TimeCamId(tid_b, j));
            covisibility += lmdb.getObservationsCountForPair(
                TimeCamId(tid_b, j), TimeCamId(tid_a, i));
        }
    }
    return covisibility;
}

// ═══════════════════════════════════════════════════════════════════
// SelectKeyframeToCull
// ═══════════════════════════════════════════════════════════════════

bool LocalMapper::SelectKeyframesToCull(std::vector<int64_t>& keyframesToCull) {
    // if (frame_poses.size() <= 5) return false;  // keep minimum map

    std::vector<int64_t> ordered;
    ordered.reserve(frame_poses.size());
    for (const auto& kv : frame_poses) {
        ordered.push_back(kv.first);
    }
    std::sort(ordered.begin(), ordered.end());
    // We want to cull the keyframes with the most redundant keyframes without
    // any preferential treatment for the latest keyframes const size_t
    // keep_recent =
    //     std::min<size_t>(mpNewKeyframesForTracking.size(), ordered.size());
    const size_t eligible_end = ordered.size();  // - keep_recent;

    const auto& obs = lmdb.getObservations();

    // Criterion 1 — redundancy.
    for (size_t i = 0; i < eligible_end; ++i) {
        const int64_t a = ordered[i];
        size_t total_a = 0;
        for (size_t cam = 0; cam < calib.intrinsics.size(); ++cam) {
            TimeCamId tcid_a(a, cam);
            auto obs_it = obs.find(tcid_a);
            if (obs_it != obs.end()) {
                for (const auto& [target, kpt_set] : obs_it->second) {
                    total_a += kpt_set.size();
                }
            }
            // total_a +=
            // lmdb.getNonLandmarkObservationsCountForKeyFrame(tcid_a);
        }
        if (total_a == 0) continue;

        // Check covisibility against every other KF.
        for (size_t j = i; j < ordered.size(); ++j) {
            if (i == j) continue;
            const int64_t b = ordered[j];
            const size_t covis =
                ComputeCovisibility(a, b, calib.intrinsics.size());
            // std::cout << "[Local Mapper] Computed covisibility: "
            //           << static_cast<double>(covis) /
            //                  static_cast<double>(total_a)
            //           << std::endl;
            if (static_cast<double>(covis) / static_cast<double>(total_a) >=
                mpCullCovisibilityThresh) {
                keyframesToCull.push_back(a);
                break;
            }
        }
    }

    // Criterion 2 — capacity (oldest eligible KF).
    if (frame_poses.size() > mpMaxLocalMapSize && keyframesToCull.empty()) {
        if (mpVioDebugMode)
            std::cout
                << "[Local Mapper] No Keyframe Found with Covisibility below "
                   "threshold; Culling Last Frame"
                << std::endl;
        keyframesToCull.push_back(ordered.front());
    }
    if (keyframesToCull.empty()) {
        return false;
    } else {
        if (mpVioDebugMode)
            std::cout << "[Local Mapper] Received " << keyframesToCull.size()
                      << " keyframes to cull" << std::endl;
        return true;
    }
}

// ═══════════════════════════════════════════════════════════════════
// FindBestRehostKf
// ═══════════════════════════════════════════════════════════════════

int64_t LocalMapper::FindBestRehostKf(int64_t culled_kf, TrackId lm_id,
                                      const std::set<int64_t>& candidates) {
    int64_t best = -1;
    size_t best_covis = 0;
    if (lm_id < 0 || !lmdb.landmarkExists(lm_id)) return -1;
    const auto& obs_set = lmdb.getLandmark(lm_id).obs;

    for (const int64_t c : candidates) {
        if (c == culled_kf) continue;
        // lm_id must be observed by c in at least one camera.
        bool observed = false;
        for (size_t cam = 0; cam < calib.intrinsics.size(); ++cam) {
            if (obs_set.count(TimeCamId(c, cam))) {
                observed = true;
                break;
            }
        }
        if (!observed) continue;

        const size_t cv =
            ComputeCovisibility(culled_kf, c, calib.intrinsics.size());
        if (cv > best_covis) {
            best_covis = cv;
            best = c;
        }
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════
// RehostLandmark
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::RehostLandmark(TrackId lm_id, int64_t culled_kf,
                                 int64_t new_host_kf) {
    const Keypoint<double>& old_kpt = lmdb.getLandmark(lm_id);

    // Preserve observations before removeLandmark tears them down.
    Eigen::aligned_map<TimeCamId, Eigen::Vector2d> obs_copy(old_kpt.obs.begin(),
                                                            old_kpt.obs.end());

    // Choose the cam_id of the new host that actually observed this landmark.
    // Prefer camera 0 for determinism when both cameras observed it.
    TimeCamId new_host_tcid(new_host_kf, 0);
    if (obs_copy.count(new_host_tcid) == 0 && calib.intrinsics.size() > 1) {
        bool found_obs = false;
        for (size_t i = 1; i < calib.intrinsics.size(); i++) {
            TimeCamId alt(new_host_kf, i);
            if (obs_copy.count(alt)) {
                new_host_tcid = alt;
                found_obs = true;
                break;
            }
        }
        if (!found_obs) return;
    } else if (obs_copy.count(new_host_tcid) == 0) {
        return;
    }

    // Compute new landmark parameters in the new host frame.
    const Eigen::Vector2d& pos_2d_new = obs_copy.at(new_host_tcid);
    Eigen::Vector4d pos_3d_new_hom;
    if (!calib.intrinsics[new_host_tcid.cam_id].unproject(pos_2d_new,
                                                          pos_3d_new_hom))
        return;

    Sophus::SE3d T_w_newh = frame_poses.at(new_host_kf).getPose() *
                            calib.T_i_c[new_host_tcid.cam_id];

    Keypoint<double> new_kpt;
    new_kpt.host_kf_id = new_host_tcid;
    bool triangulated = false;
    for (const auto& o : obs_copy) {
        if (o.first == new_host_tcid) continue;
        if (o.first.frame_id == culled_kf) continue;
        if (!frame_poses.count(o.first.frame_id)) continue;

        Eigen::Vector4d p_o_hom;
        if (!calib.intrinsics[o.first.cam_id].unproject(o.second, p_o_hom))
            continue;
        Sophus::SE3d T_w_o = frame_poses.at(o.first.frame_id).getPose() *
                             calib.T_i_c[o.first.cam_id];
        Sophus::SE3d T_newh_o = T_w_newh.inverse() * T_w_o;
        if (T_newh_o.translation().squaredNorm() <
            config.mapper_min_triangulation_dist *
                config.mapper_min_triangulation_dist)
            continue;
        Eigen::Vector4d p_3d =
            triangulate(pos_3d_new_hom.head<3>(), p_o_hom.head<3>(), T_newh_o);
        if (!p_3d.array().isFinite().all() || p_3d[3] <= 0 || p_3d[3] > 2.0)
            continue;
        new_kpt.direction = StereographicParam<double>::project(p_3d);
        new_kpt.inv_dist = p_3d[3];
        triangulated = true;
        break;
    }
    if (!triangulated) {
        lmdb.removeLandmark(lm_id);
        return;
    }

    // Perform the swap.
    lmdb.removeLandmark(lm_id);
    lmdb.addLandmark(lm_id, new_kpt);
    int obs_added = 0;
    for (const auto& o : obs_copy) {
        if (o.first.frame_id == culled_kf) continue;
        if (o.first == new_host_tcid) continue;
        KeypointObservation<double> ko;
        ko.kpt_id = lm_id;
        ko.pos = o.second;
        lmdb.addObservation(o.first, ko);
        ++obs_added;
    }

    // If every observation was either from the culled frame or from the new
    // host itself, no entry was added to the observations index for this
    // landmark. The landmark would be dangling in kpts with an empty obs set,
    // which would cause removeLandmarkHelper to receive observations.end()
    // later. Remove it immediately instead.
    if (obs_added == 0) {
        lmdb.removeLandmark(lm_id);
    }
}

// ═══════════════════════════════════════════════════════════════════
// CullRedundantKeyframes
// ═══════════════════════════════════════════════════════════════════

void LocalMapper::CullRedundantKeyframes() {
    std::vector<KeypointId> landmarksToRemove;
    std::vector<int64_t> keyframesToCull;
    bool toCull = SelectKeyframesToCull(keyframesToCull);
    if (!toCull) {
        // Still clear transient per-step data even if no culling occurred.
        img_data.clear();
        feature_tracks.clear();
        mpLatestKeyframesMatches.clear();
        return;
    }

    // ─── Step 1 — rehost landmarks hosted by culled KF ─────────────────
    for (int64_t culled : keyframesToCull) {
        std::set<int64_t> candidates;
        for (const auto& kv : frame_poses) candidates.insert(kv.first);

        // Collect unique landmark IDs hosted by the culled KF via the
        // observations index — avoids getLandmarksForHost (which throws on
        // missing keys) and produces deduplicated IDs in O(N).
        const auto& obs = lmdb.getObservations();
        std::set<KeypointId> hosted_lm_set;
        for (size_t cam = 0; cam < calib.intrinsics.size(); ++cam) {
            TimeCamId tcid(culled, cam);
            auto obs_it = obs.find(tcid);
            if (obs_it == obs.end()) continue;
            for (const auto& [target, kpt_set] : obs_it->second) {
                hosted_lm_set.insert(kpt_set.begin(), kpt_set.end());
            }
        }
        std::vector<TrackId> hosted_lm_ids(hosted_lm_set.begin(),
                                           hosted_lm_set.end());

        landmarksToRemove.clear();
        for (const TrackId lm : hosted_lm_ids) {
            const int64_t new_host = FindBestRehostKf(culled, lm, candidates);
            if (new_host < 0) {
                if (lmdb.landmarkExists(lm)) {
                    landmarksToRemove.push_back(lm);
                }
            } else {
                RehostLandmark(lm, culled, new_host);
            }
        }
        for (const KeypointId id : landmarksToRemove) {
            lmdb.removeLandmark(id);
        }

        // ─── Step 2 — remove remaining observations of culled KF ───────────
        lmdb.removeFrame(culled);

        // ─── Step 2b — drop landmarks now below min_num_obs ─────────────────
        const auto& landmarks = lmdb.getLandmarks();
        landmarksToRemove.clear();
        for (auto it = landmarks.begin(); it != landmarks.end(); ++it) {
            if (lmdb.numObservations(it->first) < 2) {
                if (lmdb.landmarkExists(it->first)) {
                    landmarksToRemove.push_back(it->first);
                }
            }
        }
        for (const KeypointId id : landmarksToRemove) {
            lmdb.removeLandmark(id);
        }

        // ─── Step 3 — prune NFR factors referencing culled KF ──────────────
        rel_pose_factors.erase(
            std::remove_if(rel_pose_factors.begin(), rel_pose_factors.end(),
                           [culled](const RelPoseFactor& f) {
                               return f.t_i_ns == culled || f.t_j_ns == culled;
                           }),
            rel_pose_factors.end());
        roll_pitch_factors.erase(
            std::remove_if(roll_pitch_factors.begin(), roll_pitch_factors.end(),
                           [culled](const RollPitchFactor& f) {
                               return f.t_ns == culled;
                           }),
            roll_pitch_factors.end());

        // ─── Step 4 — erase frame_poses and feature_corners ─────────────────
        frame_poses.erase(culled);
        for (size_t cam = 0; cam < calib.intrinsics.size(); ++cam)
            feature_corners.unsafe_erase(TimeCamId(culled, cam));

        // feature_matches: erase any entry involving culled.
        for (auto kv = feature_matches.begin(); kv != feature_matches.end();) {
            if (kv->first.first.frame_id == culled ||
                kv->first.second.frame_id == culled) {
                kv = feature_matches.unsafe_erase(kv);
            } else {
                ++kv;
            }
        }

        // ─── Step 5 — BoW database ──────────────────────────────────────────
        std::vector<TimeCamId> bow_to_drop;
        for (size_t cam = 0; cam < calib.intrinsics.size(); ++cam)
            bow_to_drop.emplace_back(TimeCamId(culled, cam));
        hash_bow_database->RemoveKeyframes(bow_to_drop);

        // ─── Step 6 — TrackBuilder cleanup ──────────────────────────────────
        mpTrackBuilder.DeleteTracksAfterCulling({culled});
    }

    // ─── Step 7 — clear transient per-step data ─────────────────────────
    if (mpVioDebugMode)
        std::cout << "[Local Mapper] clearing transient per-step data"
                  << std::endl;
    img_data.clear();
    feature_tracks.clear();
    mpLatestKeyframesMatches.clear();
}

}  // namespace basalt
