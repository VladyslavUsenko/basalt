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

#include <basalt/controller.h>
#include <basalt/serialization/headers_serialization.h>

#include <fstream>
#include <iostream>

#include "basalt/vi_estimator/vio_estimator.h"

// ABI guard, library side. NRT_EIGEN_ABI_PIN is injected by the parent
// (ros_ws/src/slam/CMakeLists.txt) and is absent in a standalone basalt build,
// so this is inert upstream and only binds inside NRT_WS.
#ifdef NRT_EIGEN_ABI_PIN
static_assert(
    EIGEN_MAX_ALIGN_BYTES == NRT_EIGEN_ABI_PIN,
    "EIGEN_MAX_ALIGN_BYTES disagrees with the pin set by the NRT parent "
    "package. This is NRT specific and relevant for standalone usage ");
#endif

namespace basalt {

Controller::Controller(const std::string& config_path,
                       const std::string& calib_path, SlamMode mode)
    : config_path_(config_path), calib_path_(calib_path), mode_(mode) {
    // Ideally we'd set queue capacities here if not default
    out_state_queue_.set_capacity(100);
}

Controller::~Controller() {
    if (!mpStoppedSlam) Stop();
}

void Controller::Stop() {
    // (1) Unblock optical flow — it will propagate nullptr to VIO.
    OpticalFlowResult::Ptr res;
    OpticalFlowInput::Ptr frame = nullptr;
    if (opt_flow_ptr_)
        res = opt_flow_ptr_->processFrame(mpCurrentFrameTime, frame);
    if (vio_estimator_ && !mpUseProducerConsumerArchitecture)
        vio_estimator_->ProcessFrame(res);

    // (2) VIO's processing loop, on receiving nullptr, pushes nullptr to
    //     out_marg_queue (= local_map_input_queue_) and out_state_queue.

    // (3) Join VIO thread explicitly (already finished by this point
    //     since VIO pushes nullptr before mapper can pop it).
    std::cout << "Stopping VIO thread" << std::endl;
    if (vio_estimator_) {
        vio_estimator_->maybe_join();
        vio_estimator_->drain_input_queues();
    }
    std::cout << "Stopped VIO thread" << std::endl;
    // (4) Local mapper receives nullptr from VIO → exits its loop.
    //     Stop() joins the mapper thread.
    std::cout << "Stopping Local Mapping" << std::endl;
    if (local_mapper_) {
        local_mapper_->Stop();
    }
    std::cout << "Local Mapping Stopped" << std::endl;

    // (5) Join optical flow via destructor.
    opt_flow_ptr_.reset();

    // (6) Drain and join the pose processing thread.
    terminate_processing_thread_ = true;
    out_state_queue_.push(nullptr);
    if (mpUseProducerConsumerArchitecture) {
        if (pose_processing_thread_.joinable()) {
            pose_processing_thread_.join();
        }
    }
    mpStoppedSlam = true;
}

void Controller::load_config() {
    // Load Config
    if (!config_path_.empty()) {
        vio_config_.load(config_path_);
    } else {
        std::cerr << "Controller: No config path provided. Using defaults."
                  << std::endl;
    }

    // Load Calibration
    std::ifstream os(calib_path_, std::ios::binary);
    if (os.is_open()) {
        cereal::JSONInputArchive archive(os);
        archive(calib_);
        std::cout << "Loaded camera with " << calib_.intrinsics.size()
                  << " cameras" << std::endl;
    } else {
        std::cerr << "Could not load camera calibration " << calib_path_
                  << std::endl;
        std::abort();
    }
}

void Controller::initialize() {
    // Default initialization with zero biases
    std::cout << "Invoke default initialisation" << std::endl;
    initialize(0, Sophus::SE3d(), Eigen::Vector3d::Zero(),
               Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
}

void Controller::initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                            const Eigen::Vector3d& vel_w_i,
                            const Eigen::Vector3d& bg,
                            const Eigen::Vector3d& ba,
                            bool useProducerConsumerArchitecture,
                            bool enableVisualisation) {
    std::cout << "Initialising SLAM with mpUseProducerConsumerArchitecture: "
              << useProducerConsumerArchitecture << std::endl;
    mpUseProducerConsumerArchitecture = useProducerConsumerArchitecture;
    std::cout << "enable visualisation: " << enableVisualisation << std::endl;
    mpEnableVisualisation = enableVisualisation;
    // 1. Create Optical Flow Frontend
    std::cout << "Setting up Optical Flow and VIO" << std::endl;
    opt_flow_ptr_ = basalt::OpticalFlowFactory::getOpticalFlow(
        vio_config_, calib_, mpUseProducerConsumerArchitecture);

    // 2. Create VIO/VO Backend
    bool use_imu = (mode_ == SlamMode::VIO);
    // Using default gravity and double precision for now, could be
    // parameterized if needed
    vio_estimator_ = basalt::VioEstimatorFactory::getVioEstimator<double>(
        vio_config_, calib_, basalt::constants::g, use_imu,
        mpUseProducerConsumerArchitecture);  // true for use_double

    // 3. Initialize the backend
    std::cout << "Initializing VIO" << std::endl;
    if (t_ns == 0 && T_w_i.log().norm() < 1e-9 && vel_w_i.norm() < 1e-9) {
        vio_estimator_->initialize(bg, ba);
    } else {
        vio_estimator_->initialize(t_ns, T_w_i, vel_w_i, bg, ba);
    }

    // 4. Wire marginalisation output to local mapper input queue.
    local_map_input_queue_.set_capacity(10);
    vio_estimator_->out_marg_queue = &local_map_input_queue_;

    // 5. Create and wire the local mapper.
    local_mapper_ = std::make_shared<basalt::LocalMapper>(calib_, vio_config_);
    local_mapper_->SetMarginalisationDataInputQueue(&local_map_input_queue_);
    local_mapper_->SetVIOPoseUpdateCallback(
        [this](const auto& poses) { vio_estimator_->QueuePoseUpdates(poses); });
    local_mapper_->Initialise();

    if (mpUseProducerConsumerArchitecture) {
        // 6. Connect Queues for producer consumer architecture
        // Connect frontend output to backend input
        opt_flow_ptr_->output_queue = &vio_estimator_->vision_data_queue;

        // Connect backend output to our output queue
        vio_estimator_->out_state_queue = &out_state_queue_;

        // Start the pose processing thread
        terminate_processing_thread_ = false;
        pose_processing_thread_ =
            std::thread(&Controller::process_pose_queue_loop, this);
    }
    std::cout << "SLAM initialisation done" << std::endl;
}

bool Controller::TrackMonocular(OpticalFlowInput::Ptr& frame, Sophus::SE3f& tcw,
                                std::optional<Sophus::SE3d> gtcw) {
    OpticalFlowResult::Ptr res =
        opt_flow_ptr_->processFrame(frame->t_ns, frame);
    mpCurrentFrameTime = frame->t_ns;
    current_latest_pose_ = vio_estimator_->ProcessFrame(res);
    // Forward the ground-truth pose to the GUI only when asked to (G1/G4).
    if (mpEnableVisualisation && mvpGroundTruthQueue && gtcw) {
        mvpGroundTruthQueue->try_push(basalt::GtPose{frame->t_ns, *gtcw});
    }
    if (current_latest_pose_) {
        tcw = current_latest_pose_->T_w_i.cast<float>();
    } else {
        return false;
    }
    return true;
}

void Controller::GrabImage(basalt::OpticalFlowInput::Ptr data) {
    if (opt_flow_ptr_) {
        opt_flow_ptr_->input_queue.push(data);
    }
}

void Controller::GrabIMU(basalt::ImuData<double>::Ptr data) {
    if (vio_estimator_) {
        vio_estimator_->imu_data_queue.push(data);
    }
}

basalt::PoseVelBiasState<double>::Ptr Controller::GetLatestPose() const {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return current_latest_pose_;
}

bool Controller::TryPopPose(basalt::PoseVelBiasState<double>::Ptr& pose) {
    return out_state_queue_.try_pop(pose);
}

void Controller::process_pose_queue_loop() {
    // TODO Will Need to check implications on latency
    basalt::PoseVelBiasState<double>::Ptr pose;
    while (!terminate_processing_thread_ || !out_state_queue_.empty()) {
        if (out_state_queue_.try_pop(pose)) {
            if (pose) {  // Check for nullptr indicating end of stream or
                         // shutdown
                std::lock_guard<std::mutex> lock(pose_mutex_);
                current_latest_pose_ = pose;
            } else {
                // nullptr received, means end of stream or shutdown signal
                break;
            }
        } else {
            // Queue empty, wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

std::shared_ptr<basalt::LocalMapper> Controller::GetLocalMapper() const {
    return local_mapper_;
}

basalt::VioEstimatorBase<double>::Ptr Controller::GetVIO() const {
    return vio_estimator_;
}

basalt::OpticalFlowBase::Ptr Controller::GetOpticalFlow() const {
    return opt_flow_ptr_;
}

basalt::Calibration<double>& Controller::GetCalibration() { return calib_; }

bool Controller::IsVisualisationEnabled() const {
    return mpEnableVisualisation;
}

void Controller::SetGroundTruthVisualisationQueue(
    tbb::concurrent_bounded_queue<basalt::GtPose>* queue) {
    mvpGroundTruthQueue = queue;
}

}  // namespace basalt
