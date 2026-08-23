#pragma once

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/local_mapper.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/visualisation/utils.h>  // basalt::GtPose

#include <basalt/calibration/calibration.hpp>
#include <mutex>
#include <optional>
#include <string>

namespace basalt {

enum class SlamMode { VO, VIO };

class Controller {
public:
    Controller(const std::string& config_path, const std::string& calib_path,
               SlamMode mode);

    ~Controller();

    // Gracefully shut down the full SLAM pipeline (OF → VIO → LocalMapper)
    // using the cascade sentinel-nullptr pattern. Safe to call multiple times.
    void Stop();

    void load_config();

    // Initialize at the start (default biases)
    void initialize();

    // Initialize mid-flight with specific state. The trailing
    // enableVisualisation flag arms the visualisation taps and is the single
    // source of truth for whether the GUI is required.
    void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                    const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                    const Eigen::Vector3d& ba,
                    bool useProducerConsumerArchitecture = false,
                    bool enableVisualisation = false);

    void GrabImage(basalt::OpticalFlowInput::Ptr data);
    void GrabIMU(basalt::ImuData<double>::Ptr data);

    // Helper to retrieve the latest pose estimate (does not pop from queue)
    basalt::PoseVelBiasState<double>::Ptr GetLatestPose() const;

    // Explicitly pop a pose reading from the queue
    bool TryPopPose(basalt::PoseVelBiasState<double>::Ptr& pose);

    // gtcw is forwarded to the GUI only when visualisation is enabled and a
    // ground-truth queue has been registered; existing callers are unaffected.
    bool TrackMonocular(OpticalFlowInput::Ptr& frame, Sophus::SE3f& tcw,
                        std::optional<Sophus::SE3d> gtcw = std::nullopt);

    std::shared_ptr<basalt::LocalMapper> GetLocalMapper() const;
    basalt::VioEstimatorBase<double>::Ptr GetVIO() const;
    basalt::OpticalFlowBase::Ptr GetOpticalFlow() const;
    basalt::Calibration<double>& GetCalibration();

    bool IsVisualisationEnabled() const;
    void SetGroundTruthVisualisationQueue(
        tbb::concurrent_bounded_queue<basalt::GtPose>* queue);

private:
    // configuration
    std::string config_path_;
    std::string calib_path_;
    SlamMode mode_;

    basalt::VioConfig vio_config_;
    basalt::Calibration<double> calib_;

    basalt::OpticalFlowBase::Ptr opt_flow_ptr_;
    basalt::VioEstimatorBase<double>::Ptr vio_estimator_;

    tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
        out_state_queue_;

    // Local mapper input queue and instance
    tbb::concurrent_bounded_queue<basalt::MargData::Ptr> local_map_input_queue_;
    std::shared_ptr<basalt::LocalMapper> local_mapper_;

    // Member for latest pose, updated by an internal thread
    basalt::PoseVelBiasState<double>::Ptr current_latest_pose_;
    mutable std::mutex pose_mutex_;
    std::thread pose_processing_thread_;
    std::atomic<bool> terminate_processing_thread_ = false;
    std::atomic<bool> mpStoppedSlam = false;

    void process_pose_queue_loop();
    bool mpUseProducerConsumerArchitecture = true;

    // Visualisation taps. Both are inert unless the GUI is enabled, so they add
    // no overhead to the estimator path when running headless.
    bool mpEnableVisualisation = false;
    tbb::concurrent_bounded_queue<basalt::GtPose>* mvpGroundTruthQueue =
        nullptr;

    int64_t mpCurrentFrameTime;
};

}  // namespace basalt
