#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <thread>

#include <librealsense2/rs.hpp>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/concurrent_queue.h>

#include <CLI/CLI.hpp>
#include <basalt/calibration/calibration.hpp>
#include <cereal/archives/json.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

constexpr int NUM_CAMS = 2;
constexpr int UI_WIDTH = 200;

pangolin::DataLog imu_log;

pangolin::Var<int> webp_quality("ui.webp_quality", 90, 0, 101);
pangolin::Var<int> skip_frames("ui.skip_frames", 1, 1, 10);

struct ImageData {
  using Ptr = std::shared_ptr<ImageData>;

  int cam_id;
  double exposure_time;
  int64_t timestamp;
  cv::Mat image;
};

struct RsIMUData {
  double timestamp;
  Eigen::Vector3d data;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

ImageData::Ptr last_images[NUM_CAMS];
tbb::concurrent_bounded_queue<ImageData::Ptr> image_save_queue;
float exposure;
std::string dataset_dir;
std::string dataset_folder;
std::string result_dir;

std::atomic<bool> stop_workers;
std::atomic<bool> record;

std::ofstream cam_data[NUM_CAMS], exposure_data[NUM_CAMS], imu0_data;

std::string get_date();

void image_save_worker() {
  ImageData::Ptr img;

  while (!stop_workers) {
    if (image_save_queue.try_pop(img)) {
#if CV_VERSION_MAJOR >= 3
      std::string filename = dataset_folder + "mav0/cam" +
                             std::to_string(img->cam_id) + "/data/" +
                             std::to_string(img->timestamp) + ".webp";

      std::vector<int> compression_params = {cv::IMWRITE_WEBP_QUALITY,
                                             webp_quality};
      cv::imwrite(filename, img->image, compression_params);
#else
      std::string filename = dataset_folder + "mav0/cam" +
                             std::to_string(img->cam_id) + "/data/" +
                             std::to_string(img->timestamp) + ".jpg";

      std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY,
                                             webp_quality};
      cv::imwrite(filename, img->image, compression_params);
#endif
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void toggle_recording() {
  record = !record;
  if (record) {
    dataset_folder = dataset_dir + "dataset_" + get_date() + "/";
    fs::create_directory(dataset_folder);
    fs::create_directory(dataset_folder + "mav0/");
    fs::create_directory(dataset_folder + "mav0/cam0/");
    fs::create_directory(dataset_folder + "mav0/cam0/data/");
    fs::create_directory(dataset_folder + "mav0/cam1/");
    fs::create_directory(dataset_folder + "mav0/cam1/data/");
    fs::create_directory(dataset_folder + "mav0/imu0/");

    cam_data[0].open(dataset_folder + "mav0/cam0/data.csv");
    cam_data[1].open(dataset_folder + "mav0/cam1/data.csv");
    exposure_data[0].open(dataset_folder + "mav0/cam0/exposure.csv");
    exposure_data[1].open(dataset_folder + "mav0/cam1/exposure.csv");
    imu0_data.open(dataset_folder + "mav0/imu0/data.csv");

    cam_data[0] << "#timestamp [ns], filename\n";
    cam_data[1] << "#timestamp [ns], filename\n";
    exposure_data[0] << "#timestamp [ns], exposure time[ns]\n";
    exposure_data[1] << "#timestamp [ns], exposure time[ns]\n";
    imu0_data << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad "
                 "s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y "
                 "[m s^-2],a_RS_S_z [m s^-2]\n";

    std::cout << "Started recording dataset in " << dataset_folder << std::endl;

  } else {
    cam_data[0].close();
    cam_data[1].close();
    exposure_data[0].close();
    exposure_data[1].close();
    imu0_data.close();

    std::cout << "Stopped recording dataset in " << dataset_folder << std::endl;
  }
}

void export_device_calibration(rs2::pipeline_profile &profile,
                               const std::string &out_path) {
  using Scalar = double;

  std::shared_ptr<basalt::Calibration<Scalar>> calib;
  calib.reset(new basalt::Calibration<Scalar>);

  auto accel_stream = profile.get_stream(RS2_STREAM_ACCEL);
  auto gyro_stream = profile.get_stream(RS2_STREAM_GYRO);
  auto cam0_stream = profile.get_stream(RS2_STREAM_FISHEYE, 1);
  auto cam1_stream = profile.get_stream(RS2_STREAM_FISHEYE, 2);

  // get gyro extrinsics
  if (auto gyro = gyro_stream.as<rs2::motion_stream_profile>()) {
    // TODO: gyro
    rs2_motion_device_intrinsic intrinsics = gyro.get_motion_intrinsics();

    std::cout << " Scale X      cross axis      cross axis  Bias X \n";
    std::cout << " cross axis    Scale Y        cross axis  Bias Y  \n";
    std::cout << " cross axis    cross axis     Scale Z     Bias Z  \n";
    for (auto &i : intrinsics.data) {
      for (int j = 0; j < 4; j++) {
        std::cout << i[j] << "    ";
      }
      std::cout << "\n";
    }

    std::cout << "Variance of noise for X, Y, Z axis \n";
    for (float noise_variance : intrinsics.noise_variances)
      std::cout << noise_variance << " ";
    std::cout << "\n";

    std::cout << "Variance of bias for X, Y, Z axis \n";
    for (float bias_variance : intrinsics.bias_variances)
      std::cout << bias_variance << " ";
    std::cout << "\n";
  } else {
    throw std::exception();
  }

  // get accel extrinsics
  if (auto gyro = accel_stream.as<rs2::motion_stream_profile>()) {
    // TODO: accel
    // rs2_motion_device_intrinsic intrinsics = accel.get_motion_intrinsics();
  } else {
    throw std::exception();
  }

  // get camera ex-/intrinsics
  for (const auto &cam_stream : {cam0_stream, cam1_stream}) {
    if (auto cam = cam_stream.as<rs2::video_stream_profile>()) {
      // extrinsics
      rs2_extrinsics ex = cam.get_extrinsics_to(gyro_stream);
      Eigen::Matrix3f rot = Eigen::Map<Eigen::Matrix3f>(ex.rotation);
      Eigen::Vector3f trans = Eigen::Map<Eigen::Vector3f>(ex.translation);

      Eigen::Quaterniond q(rot.cast<double>());
      basalt::Calibration<Scalar>::SE3 T_i_c(q, trans.cast<double>());

      std::cout << "T_i_c\n" << T_i_c.matrix() << std::endl;

      calib->T_i_c.push_back(T_i_c);

      // get resolution
      Eigen::Vector2i resolution;
      resolution << cam.width(), cam.height();
      calib->resolution.push_back(resolution);

      // intrinsics
      rs2_intrinsics intrinsics = cam.get_intrinsics();
      basalt::KannalaBrandtCamera4<Scalar>::VecN params;
      params << intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
          intrinsics.coeffs[0], intrinsics.coeffs[1], intrinsics.coeffs[2],
          intrinsics.coeffs[3];

      std::cout << "params: " << params.transpose() << std::endl;

      basalt::GenericCamera<Scalar> camera;
      basalt::KannalaBrandtCamera4 kannala_brandt(params);
      camera.variant = kannala_brandt;

      calib->intrinsics.push_back(camera);
    } else {
      throw std::exception();  // TODO: better exception
    }
  }

  // serialize and store calibration
  // std::ofstream os(out_path + "calibration.json");
  // cereal::JSONOutputArchive archive(os);

  // archive(*calib);
}

int main(int argc, char *argv[]) {
  CLI::App app{"Record RealSense T265 Data"};

  app.add_option("--dataset-dir", dataset_dir, "Path to dataset");
  app.add_option("--exposure", exposure,
                 "If set will enable manual exposure, value in microseconds.");
  app.add_option("--result-dir", result_dir,
                 "If set will enable manual exposure, value in microseconds.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  bool show_gui = true;

  image_save_queue.set_capacity(5000);

  stop_workers = false;
  std::vector<std::thread> worker_threads;
  for (int i = 0; i < 8; i++) {
    worker_threads.emplace_back(image_save_worker);
  }

  std::string color_mode;

  // realsense
  rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
  rs2::context ctx;
  rs2::pipeline pipe(ctx);
  rs2::config cfg;

  // Add streams of gyro and accelerometer to configuration
  cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
  cfg.enable_stream(RS2_STREAM_FISHEYE, 1, RS2_FORMAT_Y8);
  cfg.enable_stream(RS2_STREAM_FISHEYE, 2, RS2_FORMAT_Y8);

  // Using the device_hub we can block the program until a device connects
  //  rs2::device_hub device_hub(ctx);

  auto devices = ctx.query_devices();
  if (devices.size() == 0) {
    std::abort();
  }
  auto device = devices[0];
  std::cout << "Device " << device.get_info(RS2_CAMERA_INFO_NAME)
            << " connected" << std::endl;
  auto sens = device.query_sensors()[0];

  if (exposure > 0) {
    std::cout << "Setting exposure to: " << exposure << " microseconds"
              << std::endl;
    sens.set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
    sens.set_option(rs2_option::RS2_OPTION_EXPOSURE, (float)exposure);
  }

  std::mutex data_mutex;

  rs2::motion_frame last_gyro_meas = rs2::motion_frame(rs2::frame());
  Eigen::deque<RsIMUData> gyro_data_queue;

  std::shared_ptr<RsIMUData> prev_accel_data;

  int processed_frame = 0;

  auto callback = [&](const rs2::frame &frame) {
    std::lock_guard<std::mutex> lock(data_mutex);

    if (auto fp = frame.as<rs2::motion_frame>()) {
      auto motion = frame.as<rs2::motion_frame>();

      if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO &&
          motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
        RsIMUData d;
        d.timestamp = motion.get_timestamp();
        d.data << motion.get_motion_data().x, motion.get_motion_data().y,
            motion.get_motion_data().z;

        gyro_data_queue.emplace_back(d);
      }

      if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL &&
          motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
        RsIMUData d;
        d.timestamp = motion.get_timestamp();
        d.data << motion.get_motion_data().x, motion.get_motion_data().y,
            motion.get_motion_data().z;

        if (!prev_accel_data.get()) {
          prev_accel_data.reset(new RsIMUData(d));
        } else {
          BASALT_ASSERT(d.timestamp > prev_accel_data->timestamp);

          while (!gyro_data_queue.empty() && gyro_data_queue.front().timestamp <
                                                 prev_accel_data->timestamp) {
            std::cout << "Skipping gyro data. Timestamp before the first accel "
                         "measurement.";
            gyro_data_queue.pop_front();
          }

          while (!gyro_data_queue.empty() &&
                 gyro_data_queue.front().timestamp < d.timestamp) {
            RsIMUData gyro_data = gyro_data_queue.front();
            gyro_data_queue.pop_front();

            double w0 = (d.timestamp - gyro_data.timestamp) /
                        (d.timestamp - prev_accel_data->timestamp);

            double w1 = (gyro_data.timestamp - prev_accel_data->timestamp) /
                        (d.timestamp - prev_accel_data->timestamp);

            Eigen::Vector3d accel_interpolated =
                w0 * prev_accel_data->data + w1 * d.data;

            if (record) {
              int64_t timestamp = gyro_data.timestamp * 1e6;
              imu0_data << timestamp << "," << gyro_data.data[0] << ","
                        << gyro_data.data[1] << "," << gyro_data.data[2] << ","
                        << accel_interpolated[0] << "," << accel_interpolated[1]
                        << "," << accel_interpolated[2] << "\n";
            }

            imu_log.Log(accel_interpolated[0], accel_interpolated[1],
                        accel_interpolated[2]);
          }

          prev_accel_data.reset(new RsIMUData(d));
        }
      }
    }

    if (auto fs = frame.as<rs2::frameset>()) {
      processed_frame++;
      if (processed_frame % int(skip_frames) != 0) return;

      for (int i = 0; i < NUM_CAMS; i++) {
        auto f = fs[i];
        if (!f.as<rs2::video_frame>()) {
          std::cout << "Weird Frame, skipping" << std::endl;
          continue;
        }
        auto vf = f.as<rs2::video_frame>();

        last_images[i].reset(new ImageData);
        last_images[i]->image = cv::Mat(vf.get_height(), vf.get_width(), CV_8U);
        std::memcpy(
            last_images[i]->image.ptr(), vf.get_data(),
            vf.get_width() * vf.get_height() * vf.get_bytes_per_pixel());

        last_images[i]->exposure_time =
            vf.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE);

        last_images[i]->timestamp = vf.get_timestamp() * 1e6;
        last_images[i]->cam_id = i;

        if (record) {
          image_save_queue.push(last_images[i]);

          cam_data[i] << last_images[i]->timestamp << ","
                      << last_images[i]->timestamp << ".webp" << std::endl;

          exposure_data[i] << last_images[i]->timestamp << ","
                           << int64_t(vf.get_frame_metadata(
                                          RS2_FRAME_METADATA_ACTUAL_EXPOSURE) *
                                      1e3)
                           << std::endl;
        }
      }
    }
  };

  // Start streaming through the callback
  rs2::pipeline_profile profiles = pipe.start(cfg, callback);

  {
    auto sensors = profiles.get_device().query_sensors();

    for (auto &s : sensors) {
      std::cout << "Sensor " << s.get_info(RS2_CAMERA_INFO_NAME)
                << ". Supported options:" << std::endl;

      for (const auto &o : s.get_supported_options()) {
        std::cout << "\t" << rs2_option_to_string(o) << std::endl;
      }
    }
  }

  export_device_calibration(profiles, result_dir);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Record RealSense T265", 1200, 800);

    pangolin::Var<std::function<void(void)>> record_btn("ui.record",
                                                        toggle_recording);

    std::atomic<int64_t> record_t_ns;
    record_t_ns = 0;

    glEnable(GL_DEPTH_TEST);

    pangolin::View &img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::View &plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      int idx = img_view.size();
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      iv->extern_draw_function = [&, idx](pangolin::View &v) {
        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);  // red
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (last_images[idx].get())
          pangolin::GlFont::I()
              .Text("Exposure: %.3f ms.",
                    last_images[idx]->exposure_time / 1000.0)
              .Draw(30, 30);

        if (idx == 0) {
          pangolin::GlFont::I()
              .Text("Queue: %d.", image_save_queue.size())
              .Draw(30, 60);
        }

        if (idx == 0 && record) {
          pangolin::GlFont::I().Text("Recording").Draw(30, 90);
        }
      };

      iv->OnSelectionCallback =
          [&](pangolin::ImageView::OnSelectionEventData o) {
            int64_t curr_t_ns = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();
            if (std::abs(record_t_ns - curr_t_ns) > int64_t(2e9)) {
              toggle_recording();
              record_t_ns = curr_t_ns;
            }
          };

      img_view.push_back(iv);
      img_view_display.AddDisplay(*iv);
    }

    imu_log.Clear();

    std::vector<std::string> labels;
    labels.push_back(std::string("accel x"));
    labels.push_back(std::string("accel y"));
    labels.push_back(std::string("accel z"));
    imu_log.SetLabels(labels);

    pangolin::Plotter plotter(&imu_log, 0.0f, 2000.0f, -15.0f, 15.0f, 0.1f,
                              0.1f);
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
    plotter.Track("$i");

    plot_display.AddDisplay(plotter);

    plotter.ClearSeries();
    plotter.AddSeries("$i", "$0", pangolin::DrawingModeLine,
                      pangolin::Colour::Red(), "accel x");
    plotter.AddSeries("$i", "$1", pangolin::DrawingModeLine,
                      pangolin::Colour::Green(), "accel y");
    plotter.AddSeries("$i", "$2", pangolin::DrawingModeLine,
                      pangolin::Colour::Blue(), "accel z");

    while (!pangolin::ShouldQuit()) {
      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_BYTE;
        fmt.scalable_internal_format = GL_LUMINANCE8;

        for (size_t cam_id = 0; cam_id < NUM_CAMS; cam_id++) {
          if (last_images[cam_id].get())
            img_view[cam_id]->SetImage(last_images[cam_id]->image.ptr(),
                                       last_images[cam_id]->image.cols,
                                       last_images[cam_id]->image.rows,
                                       last_images[cam_id]->image.step, fmt);
        }
      }

      pangolin::FinishFrame();

      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
  }

  stop_workers = true;
  for (auto &t : worker_threads) {
    t.join();
  }

  return EXIT_SUCCESS;
}

std::string get_date() {
  constexpr int MAX_DATE = 64;
  time_t now;
  char the_date[MAX_DATE];

  the_date[0] = '\0';

  now = time(NULL);

  if (now != -1) {
    strftime(the_date, MAX_DATE, "%Y_%m_%d_%H_%M_%S", gmtime(&now));
  }

  return std::string(the_date);
}
