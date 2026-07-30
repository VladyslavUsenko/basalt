/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.
*/

#include <basalt/utils/vis_utils.h>  // render_camera, getcolor, colour palette
#include <basalt/visualisation/visualiser.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>

#include <functional>

namespace basalt {

SlamVisualiser::SlamVisualiser(basalt::Controller& controller)
    : mpController(controller),
      mpCalib(controller.GetCalibration()),
      mpVio(controller.GetVIO()),
      mpLocalMapper(controller.GetLocalMapper()),
      mpOpticalFlow(controller.GetOpticalFlow()) {}

SlamVisualiser::~SlamVisualiser() { Stop(); }

// ═══════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════

void SlamVisualiser::Start() {
    if (!mpController.IsVisualisationEnabled()) return;  // honour the gate (G4)

    // Bounded capacities; producers use try_push so a full queue drops the
    // newest frame instead of blocking the estimator threads (G3).
    mvpVioVisQueue.set_capacity(20);
    mvpVioStateQueue.set_capacity(100);
    mvpLocalMapVisQueue.set_capacity(4);
    mvpGroundTruthQueue.set_capacity(200);
    {
        std::lock_guard<std::mutex> lock(mpMtxQuitVisualiser);
        mpQuitVisualiser = false;
    }

    pangolin::CreateWindowAndBind("Basalt SLAM", 1800, 1000);
    glEnable(GL_DEPTH_TEST);  // per-context state; must follow window creation
    SetupLayout();

    // Connect the visualiser's queues to the estimator output hooks. These are
    // raw-pointer taps, identical to how src/vio.cpp connects out_vis_queue.
    mpVio->out_vis_queue = &mvpVioVisQueue;
    mpVio->out_state_queue = &mvpVioStateQueue;
    mpLocalMapper->out_vis_queue = &mvpLocalMapVisQueue;
    mpController.SetGroundTruthVisualisationQueue(&mvpGroundTruthQueue);

    mpRunning = true;
    mpVioVisConsumerThread =
        std::thread(&SlamVisualiser::ConsumeVioVisQueue, this);
    mpVioStateConsumerThread =
        std::thread(&SlamVisualiser::ConsumeVioStateQueue, this);
    mpLocalMapConsumerThread =
        std::thread(&SlamVisualiser::ConsumeLocalMapQueue, this);
}

void SlamVisualiser::Stop() {
    {
        std::lock_guard<std::mutex> lock(mpMtxQuitVisualiser);
        mpQuitVisualiser = true;
    }

    if (!mpRunning.exchange(false)) return;  // idempotent

    // Release the blocking pop()s with the cascade nullptr sentinel (P3).
    mvpVioVisQueue.push(nullptr);
    mvpVioStateQueue.push(nullptr);
    mvpLocalMapVisQueue.push(nullptr);

    if (mpVioVisConsumerThread.joinable()) mpVioVisConsumerThread.join();
    if (mpVioStateConsumerThread.joinable()) mpVioStateConsumerThread.join();
    if (mpLocalMapConsumerThread.joinable()) mpLocalMapConsumerThread.join();
}

// ═══════════════════════════════════════════════════════════════════
// Layout
// ═══════════════════════════════════════════════════════════════════

void SlamVisualiser::SetupLayout() {
    constexpr int UI_WIDTH = 200;

    pangolin::View& main_display = pangolin::CreateDisplay().SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::View& img_view_display = pangolin::CreateDisplay()
                                           .SetBounds(0.4, 1.0, 0.0, 0.4)
                                           .SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    mpPlotter = new pangolin::Plotter(&mpVioDataLog, 0.0, 100, -10.0, 10.0,
                                      0.01f, 0.01f);
    plot_display.AddDisplay(*mpPlotter);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // Toggles. Each Var name carries the "ui." prefix so the panel auto-renders
    // it. They must be constructed here, after the context and panel exist,
    // which is why they are held by unique_ptr rather than at file scope.
    mpShowObs =
        std::make_unique<pangolin::Var<bool>>("ui.show_obs", true, false, true);
    mpShowFlow = std::make_unique<pangolin::Var<bool>>("ui.show_flow", false,
                                                       false, true);
    mpShowIds = std::make_unique<pangolin::Var<bool>>("ui.show_ids", false,
                                                      false, true);
    mpShowGt =
        std::make_unique<pangolin::Var<bool>>("ui.show_gt", true, false, true);
    mpShowEstPos = std::make_unique<pangolin::Var<bool>>("ui.show_est_pos",
                                                         true, false, true);
    mpShowEstVel = std::make_unique<pangolin::Var<bool>>("ui.show_est_vel",
                                                         false, false, true);
    mpShowEstBg = std::make_unique<pangolin::Var<bool>>("ui.show_est_bg", false,
                                                        false, true);
    mpShowEstBa = std::make_unique<pangolin::Var<bool>>("ui.show_est_ba", false,
                                                        false, true);
    mpFollow =
        std::make_unique<pangolin::Var<bool>>("ui.follow", true, false, true);
    mpShowLocalMapPoints = std::make_unique<pangolin::Var<bool>>(
        "ui.show_local_map_points", true, false, true);
    mpShowLocalMapKfs = std::make_unique<pangolin::Var<bool>>(
        "ui.show_local_map_kfs", true, false, true);

    // One ImageView per camera, each bound to DrawImageOverlay with its camera
    // index captured via a member-function bind on this.
    while (mpImgViews.size() < mpCalib.intrinsics.size()) {
        auto iv = std::make_shared<pangolin::ImageView>();
        size_t idx = mpImgViews.size();
        mpImgViews.push_back(iv);
        img_view_display.AddDisplay(*iv);
        iv->extern_draw_function = std::bind(&SlamVisualiser::DrawImageOverlay,
                                             this, std::placeholders::_1, idx);
    }

    // Initial virtual-camera placement reuses the vio.cpp recipe: a canonical
    // offset rotated into the world frame by the VIO's initial pose.
    Eigen::Vector3d cam_p(-0.5, -3, -5);
    cam_p = mpVio->getT_w_i_init().so3() * mpCalib.T_i_c[0].so3() * cam_p;
    mpCamera = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0,
                                  pangolin::AxisZ));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.4, 1.0, 0.4, 1.0)
            .SetHandler(new pangolin::Handler3D(mpCamera));
    display3D.extern_draw_function =
        std::bind(&SlamVisualiser::DrawScene, this, std::placeholders::_1);

    main_display.AddDisplay(img_view_display);
    main_display.AddDisplay(display3D);
}

// ═══════════════════════════════════════════════════════════════════
// Render loop (main GL thread only)
// ═══════════════════════════════════════════════════════════════════

void SlamVisualiser::Run() {
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Drain ground truth on the GL thread, so this buffer has a single
        // writer and needs no lock.
        // Snapshot under lock: ConsumeVioStateQueue writes mpSlamFirstPose from
        // another thread.
        std::optional<Sophus::SE3d> slam_first;
        {
            std::lock_guard<std::mutex> lock(mpMtxVioState);
            slam_first = mpSlamFirstPose;
        }

        basalt::GtPose gt;
        while (mvpGroundTruthQueue.try_pop(gt)) {
            // IO-normalised GT is in the initial IMU body frame. The SLAM
            // gravity-aligns its world frame so its first T_w_i != Identity.
            // Multiplying by slam_first maps GT into the SLAM world frame.
            Sophus::SE3d T_gt_vis =
                slam_first ? *slam_first * gt.T_w_i : gt.T_w_i;
            mvpGroundTruthTrajectory.emplace_back(T_gt_vis.translation());
        }

        basalt::VioVisualizationData::Ptr latest;
        {
            std::lock_guard<std::mutex> lock(mpMtxVioVis);
            latest = mpLatestVio;
        }

        // Follow-camera: re-anchor to the latest body pose with rotation zeroed
        // so the world stays upright.
        if (*mpFollow && latest) {
            Sophus::SE3d T_w_i;
            if (!latest->states.empty())
                T_w_i = latest->states.back();
            else if (!latest->frames.empty())
                T_w_i = latest->frames.back();
            T_w_i.so3() = Sophus::SO3d();
            mpCamera.Follow(T_w_i.matrix());
        }

        // Upload the current frame's images once per new timestamp, so the GPU
        // texture refreshes only when a new frame actually arrives.
        if (latest && latest->opt_flow_res &&
            latest->opt_flow_res->input_images &&
            latest->t_ns != mpLastImageTns) {
            const auto& imgs = latest->opt_flow_res->input_images->img_data;
            pangolin::GlPixFormat fmt;
            fmt.glformat = GL_LUMINANCE;
            fmt.gltype = GL_UNSIGNED_SHORT;
            fmt.scalable_internal_format = GL_LUMINANCE16;
            for (size_t cam_id = 0;
                 cam_id < mpImgViews.size() && cam_id < imgs.size(); cam_id++) {
                if (imgs[cam_id].img.get())
                    mpImgViews[cam_id]->SetImage(
                        imgs[cam_id].img->ptr, imgs[cam_id].img->w,
                        imgs[cam_id].img->h, imgs[cam_id].img->pitch, fmt);
            }
            mpLastImageTns = latest->t_ns;
        }

        if (mpShowEstPos->GuiChanged() || mpShowEstVel->GuiChanged() ||
            mpShowEstBg->GuiChanged() || mpShowEstBa->GuiChanged())
            DrawPlots();

        pangolin::FinishFrame();  // renders the view tree, fires callbacks,
                                  // swaps
        {
            std::lock_guard<std::mutex> lock(mpMtxQuitVisualiser);
            if (mpQuitVisualiser) break;
        }
    }

    pangolin::Quit();

    mpImgViews.clear();
    delete mpPlotter;
    mpPlotter = nullptr;

    pangolin::DestroyWindow("Basalt SLAM");
}

// ═══════════════════════════════════════════════════════════════════
// Draw callbacks (main GL thread only)
// ═══════════════════════════════════════════════════════════════════

void SlamVisualiser::DrawScene(pangolin::View& view) {
    view.Activate(mpCamera);  // glViewport + upload projection/view matrices
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Copy each cache under its own lock; never hold two locks at once and
    // never do GL work while copying, so the consumer threads stay unblocked.
    basalt::VioVisualizationData::Ptr vio;
    basalt::LocalMapperVisualizationData::Ptr lm;
    {
        std::lock_guard<std::mutex> lock(mpMtxVioVis);
        vio = mpLatestVio;
    }
    {
        std::lock_guard<std::mutex> lock(mpMtxLocalMap);
        lm = mpLatestLocalMap;
    }

    // ── 1. VIO estimated trajectory (red) ─────────────────────────────
    // Held under the state lock because the consumer may emplace_back and
    // reallocate the buffer while the GL thread iterates it.
    {
        std::lock_guard<std::mutex> lock(mpMtxVioState);
        glColor3ubv(cam_color);
        if (!mvpVioTrajectory.empty())
            pangolin::glDrawLineStrip(mvpVioTrajectory);
    }

    // ── 2. Ground-truth trajectory (green) ────────────────────────────
    if (*mpShowGt) {
        glColor3ubv(gt_color);
        pangolin::glDrawLineStrip(mvpGroundTruthTrajectory);
    }

    // ── 3. Latest VIO frusta + sliding-window landmarks ───────────────
    if (vio) {
        for (const auto& p : vio->states)
            for (size_t i = 0; i < mpCalib.T_i_c.size(); i++)
                render_camera((p * mpCalib.T_i_c[i]).matrix(), 2.0f,
                              state_color, 0.1f);
        for (const auto& p : vio->frames)
            for (size_t i = 0; i < mpCalib.T_i_c.size(); i++)
                render_camera((p * mpCalib.T_i_c[i]).matrix(), 2.0f, pose_color,
                              0.1f);
        glPointSize(3);
        glColor3ubv(pose_color);
        pangolin::glDrawPoints(vio->points);
    }

    // ── 4. Local map: distinct marker (orange, larger) + amber KF frusta ─
    if (lm) {
        if (*mpShowLocalMapPoints) {
            glPointSize(3);
            glColor3ubv(local_map_point_color);
            pangolin::glDrawPoints(lm->points);
        }
        if (*mpShowLocalMapKfs)
            for (const auto& kf : lm->keyframes)
                for (size_t i = 0; i < mpCalib.T_i_c.size(); i++)
                    render_camera((kf * mpCalib.T_i_c[i]).matrix(), 2.0f,
                                  local_map_kf_color, 0.1f);
    }

    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);  // world origin triad
}

void SlamVisualiser::DrawImageOverlay(pangolin::View& v, size_t cam_id) {
    (void)v;

    basalt::VioVisualizationData::Ptr vio;
    {
        std::lock_guard<std::mutex> lock(mpMtxVioVis);
        vio = mpLatestVio;
    }
    if (!vio) return;

    if (*mpShowObs && cam_id < vio->projections.size()) {
        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const auto& points = vio->projections[cam_id];
        if (points.size() > 0) {
            // Depth range over all cameras drives the hue ramp so colours are
            // comparable across the stereo grid.
            double min_id = points[0][2], max_id = points[0][2];
            for (const auto& points2 : vio->projections)
                for (const auto& p : points2) {
                    min_id = std::min(min_id, p[2]);
                    max_id = std::max(max_id, p[2]);
                }

            for (const auto& c : points) {
                const float radius = 6.5;
                float r, g, b;
                getcolor(c[2] - min_id, max_id - min_id, b, g, r);
                glColor3f(r, g, b);
                pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

                if (*mpShowIds)
                    pangolin::GlFont::I()
                        .Text("%d", int(c[3]))
                        .Draw(c[0], c[1]);
            }
        }

        glColor3f(1.0, 0.0, 0.0);
        pangolin::GlFont::I()
            .Text("Tracked %d points", points.size())
            .Draw(5, 20);
    }

    if (*mpShowFlow && vio->opt_flow_res && mpOpticalFlow &&
        cam_id < vio->opt_flow_res->observations.size()) {
        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const Eigen::aligned_map<basalt::KeypointId, Eigen::AffineCompact2f>&
            kp_map = vio->opt_flow_res->observations[cam_id];

        for (const auto& kv : kp_map) {
            Eigen::MatrixXf transformed_patch =
                kv.second.linear() * mpOpticalFlow->patch_coord;
            transformed_patch.colwise() += kv.second.translation();

            for (int i = 0; i < transformed_patch.cols(); i++) {
                const Eigen::Vector2f c = transformed_patch.col(i);
                pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
            }

            const Eigen::Vector2f c = kv.second.translation();
            if (*mpShowIds)
                pangolin::GlFont::I()
                    .Text("%d", kv.first)
                    .Draw(5 + c[0], 5 + c[1]);
        }

        pangolin::GlFont::I()
            .Text("%d opt_flow patches", kp_map.size())
            .Draw(5, 20);
    }
}

void SlamVisualiser::DrawPlots() {
    mpPlotter->ClearSeries();
    mpPlotter->ClearMarkers();

    // Column layout matches ConsumeVioStateQueue: $0 time, $1-$3 velocity,
    // $4-$6 position, $7-$9 gyro bias, $10-$12 accel bias.
    if (*mpShowEstPos) {
        mpPlotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                             pangolin::Colour::Red(), "position x",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                             pangolin::Colour::Green(), "position y",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                             pangolin::Colour::Blue(), "position z",
                             &mpVioDataLog);
    }

    if (*mpShowEstVel) {
        mpPlotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                             pangolin::Colour::Red(), "velocity x",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                             pangolin::Colour::Green(), "velocity y",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                             pangolin::Colour::Blue(), "velocity z",
                             &mpVioDataLog);
    }

    if (*mpShowEstBg) {
        mpPlotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                             pangolin::Colour::Red(), "gyro bias x",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                             pangolin::Colour::Green(), "gyro bias y",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                             pangolin::Colour::Blue(), "gyro bias z",
                             &mpVioDataLog);
    }

    if (*mpShowEstBa) {
        mpPlotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                             pangolin::Colour::Red(), "accel bias x",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                             pangolin::Colour::Green(), "accel bias y",
                             &mpVioDataLog);
        mpPlotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                             pangolin::Colour::Blue(), "accel bias z",
                             &mpVioDataLog);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Consumer threads (no GL calls; fill caches only)
// ═══════════════════════════════════════════════════════════════════

void SlamVisualiser::ConsumeVioVisQueue() {
    basalt::VioVisualizationData::Ptr data;
    while (true) {
        mvpVioVisQueue.pop(data);  // blocking
        if (!data) break;          // nullptr sentinel from Stop()
        std::lock_guard<std::mutex> lock(mpMtxVioVis);
        mpLatestVio = data;  // live edge only
    }
}

void SlamVisualiser::ConsumeVioStateQueue() {
    basalt::PoseVelBiasState<double>::Ptr data;
    while (true) {
        mvpVioStateQueue.pop(data);
        if (!data) break;
        if (mpStartTns < 0) mpStartTns = data->t_ns;

        {
            std::lock_guard<std::mutex> lock(mpMtxVioState);
            if (!mpSlamFirstPose) mpSlamFirstPose = data->T_w_i;
            mvpVioTrajectory.emplace_back(data->T_w_i.translation());
        }

        // DataLog is internally mutex-protected, so logging here while the
        // plotter reads on the GL thread is safe.
        std::vector<float> vals;
        vals.push_back((data->t_ns - mpStartTns) * 1e-9);
        for (int i = 0; i < 3; i++) vals.push_back(data->vel_w_i[i]);
        for (int i = 0; i < 3; i++)
            vals.push_back(data->T_w_i.translation()[i]);
        for (int i = 0; i < 3; i++) vals.push_back(data->bias_gyro[i]);
        for (int i = 0; i < 3; i++) vals.push_back(data->bias_accel[i]);
        mpVioDataLog.Log(vals);
    }
}

void SlamVisualiser::ConsumeLocalMapQueue() {
    basalt::LocalMapperVisualizationData::Ptr data;
    while (true) {
        mvpLocalMapVisQueue.pop(data);
        if (!data) break;
        std::lock_guard<std::mutex> lock(mpMtxLocalMap);
        mpLatestLocalMap = data;  // keep only the newest map
    }
}

}  // namespace basalt
