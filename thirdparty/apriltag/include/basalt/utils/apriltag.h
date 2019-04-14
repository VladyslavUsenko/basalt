
#include <basalt/utils/sophus_utils.hpp>
#include <basalt/utils/image.h>
#include <vector>


namespace basalt {

struct ApriltagDetectorData;

class ApriltagDetector {
 public:
  ApriltagDetector();

  ~ApriltagDetector();

  void detectTags(basalt::ManagedImage<uint16_t>& img_raw,
                  Eigen::vector<Eigen::Vector2d>& corners,
                  std::vector<int>& ids,
                  std::vector<double>& radii,
                  Eigen::vector<Eigen::Vector2d>& corners_rejected,
                  std::vector<int>& ids_rejected,
                  std::vector<double>& radii_rejected);

  private:
  ApriltagDetectorData* data;
 };

}

