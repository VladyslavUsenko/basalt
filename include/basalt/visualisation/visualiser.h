#pragma once

#include <pangolin/pangolin.h>
// pangolin.h does not pull in the display widgets, so ImageView (a member type
// below) must be included explicitly to keep this header self-sufficient.
#include <basalt/controller.h>
#include <basalt/vi_estimator/vio_estimator.h>  // VioVisualizationData
#include <basalt/visualisation/utils.h>  // LocalMapperVisualizationData, GtPose
#include <pangolin/display/image_view.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace basalt {

// Unified real-time visualiser for the basalt_slam integration test. It renders
// the live VIO trajectory (image grid, sliding-window landmarks, keyframes,
// state plotter) and the live local map (mapped keyframes and triangulated
// landmarks) in a single Pangolin window. All Pangolin and OpenGL state lives
// here so the core basalt library remains Pangolin-free.
class SlamVisualiser {
public:
    // Captures references to the Controller's VIO, local mapper and
    // calibration. No GL work happens in the constructor so it may run off the
    // main thread.
    explicit SlamVisualiser(basalt::Controller& controller);
    ~SlamVisualiser();

    // Create the GL window and layout, wire the queues to the estimators, and
    // start the consumer threads. MUST be called on the main (GL) thread.
    void Start();

    // Blocking main-thread render loop; returns when the window is closed.
    void Run();

    // Sentinel-nullptr shutdown of the consumer threads; idempotent.
    void Stop();

private:
    // ── draw callbacks (bound into Pangolin extern_draw_function) ──────
    void DrawScene(pangolin::View& view);
    void DrawImageOverlay(pangolin::View& view, size_t cam_id);
    void DrawPlots();
    void SetupLayout();

    // ── consumer-thread bodies (no GL calls) ──────────────────────────
    void ConsumeVioVisQueue();    // → mpLatestVio
    void ConsumeVioStateQueue();  // → mpVioDataLog + mvpVioTrajectory
    void ConsumeLocalMapQueue();  // → mpLatestLocalMap

    // ── references into the Controller ────────────────────────────────
    basalt::Controller& mpController;
    basalt::Calibration<double>& mpCalib;
    basalt::VioEstimatorBase<double>::Ptr mpVio;
    std::shared_ptr<basalt::LocalMapper> mpLocalMapper;
    // Held only to read patch_coord for the optical-flow overlay (G1).
    basalt::OpticalFlowBase::Ptr mpOpticalFlow;

    // ── queues owned here, connected to the estimators in Start() ─────
    tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr>
        mvpVioVisQueue;
    tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
        mvpVioStateQueue;
    tbb::concurrent_bounded_queue<basalt::LocalMapperVisualizationData::Ptr>
        mvpLocalMapVisQueue;
    tbb::concurrent_bounded_queue<basalt::GtPose> mvpGroundTruthQueue;

    // ── latest-only caches (live edge; no history, no image cache) ────
    basalt::VioVisualizationData::Ptr mpLatestVio;
    basalt::LocalMapperVisualizationData::Ptr mpLatestLocalMap;
    std::mutex mpMtxVioVis;    // guards mpLatestVio
    std::mutex mpMtxVioState;  // guards mvpVioTrajectory
    std::mutex mpMtxLocalMap;  // guards mpLatestLocalMap
    std::mutex mpMtxQuitVisualiser;

    bool mpQuitVisualiser = false;

    // ── trajectories: positions only, cheap to retain ────────────────
    Eigen::aligned_vector<Eigen::Vector3d> mvpVioTrajectory;
    Eigen::aligned_vector<Eigen::Vector3d>
        mvpGroundTruthTrajectory;  // main-thread only

    // First SLAM pose (gravity-aligned T_w_i from VIO init). Applied to every
    // IO-normalised GT pose so both trajectories share the same world frame.
    std::optional<Sophus::SE3d> mpSlamFirstPose;

    // ── Pangolin objects ──────────────────────────────────────────────
    pangolin::OpenGlRenderState mpCamera;
    pangolin::DataLog mpVioDataLog;
    pangolin::Plotter* mpPlotter = nullptr;
    std::vector<std::shared_ptr<pangolin::ImageView>> mpImgViews;
    int64_t mpStartTns = -1;      // x-axis origin for the plotter
    int64_t mpLastImageTns = -1;  // dedupes image-grid uploads

    // ── toggles (registered into the "ui." panel) ─────────────────────
    std::unique_ptr<pangolin::Var<bool>> mpShowObs, mpShowFlow, mpShowIds,
        mpShowGt, mpShowEstPos, mpShowEstVel, mpShowEstBg, mpShowEstBa,
        mpFollow, mpShowLocalMapPoints, mpShowLocalMapKfs;

    // ── consumer threads + lifecycle flags ────────────────────────────
    std::thread mpVioVisConsumerThread, mpVioStateConsumerThread,
        mpLocalMapConsumerThread;
    std::atomic<bool> mpRunning{false};
};

}  // namespace basalt
