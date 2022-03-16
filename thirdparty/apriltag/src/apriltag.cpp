
#include <basalt/utils/apriltag.h>

#include <apriltags/TagDetector.h>

#include <apriltags/Tag36h11.h>

namespace basalt {

struct ApriltagDetectorData {
  ApriltagDetectorData(int numTags)
      : doSubpixRefinement(true),
        maxSubpixDisplacement(0),
        minTagsForValidObs(4),
        minBorderDistance(4.0),
        blackTagBorder(2),
        _tagCodes(AprilTags::tagCodes36h11),
        _numTags(numTags) {
    _tagDetector =
        std::make_shared<AprilTags::TagDetector>(_tagCodes, blackTagBorder);
  }

  bool doSubpixRefinement;
  double
      maxSubpixDisplacement;  //!< maximum displacement for subpixel refinement.
                              //!< If 0, only base it on tag size.
  unsigned int minTagsForValidObs;
  double minBorderDistance;
  unsigned int blackTagBorder;

  AprilTags::TagCodes _tagCodes;
  std::shared_ptr<AprilTags::TagDetector> _tagDetector;

  int _numTags;  //!< number of tags in the grid (determines the valid ids)

  inline int size() { return _numTags * 4; }
};

ApriltagDetector::ApriltagDetector(int numTags) {
  data = new ApriltagDetectorData(numTags);
}

ApriltagDetector::~ApriltagDetector() { delete data; }

void ApriltagDetector::detectTags(
    basalt::ManagedImage<uint16_t>& img_raw,
    Eigen::aligned_vector<Eigen::Vector2d>& corners, std::vector<int>& ids,
    std::vector<double>& radii,
    Eigen::aligned_vector<Eigen::Vector2d>& corners_rejected,
    std::vector<int>& ids_rejected, std::vector<double>& radii_rejected) {
  corners.clear();
  ids.clear();
  radii.clear();
  corners_rejected.clear();
  ids_rejected.clear();
  radii_rejected.clear();

  cv::Mat image(img_raw.h, img_raw.w, CV_8U);

  uint8_t* dst = image.ptr();
  const uint16_t* src = img_raw.ptr;

  for (size_t i = 0; i < img_raw.size(); i++) {
    dst[i] = (src[i] >> 8);
  }

  // detect the tags
  std::vector<AprilTags::TagDetection> detections =
      data->_tagDetector->extractTags(image);

  /* handle the case in which a tag is identified but not all tag
   * corners are in the image (all data bits in image but border
   * outside). tagCorners should still be okay as apriltag-lib
   * extrapolates them, only the subpix refinement will fail
   */

  // min. distance [px] of tag corners from image border (tag is not used if
  // violated)
  std::vector<AprilTags::TagDetection>::iterator iter = detections.begin();
  for (iter = detections.begin(); iter != detections.end();) {
    // check all four corners for violation
    bool remove = false;

    for (int j = 0; j < 4; j++) {
      remove |= iter->p[j].first < data->minBorderDistance;
      remove |= iter->p[j].first >
                (float)(image.cols) - data->minBorderDistance;  // width
      remove |= iter->p[j].second < data->minBorderDistance;
      remove |= iter->p[j].second >
                (float)(image.rows) - data->minBorderDistance;  // height
    }

    // also remove tags that are flagged as bad
    if (iter->good != 1) remove |= true;

    // also remove if the tag ID is out-of-range for this grid (faulty
    // detection)
    if (iter->id >= (int)data->size() / 4) remove |= true;

    // delete flagged tags
    if (remove) {
      // delete the tag and advance in list
      iter = detections.erase(iter);
    } else {
      // advance in list
      ++iter;
    }
  }

  // did we find enough tags?
  if (detections.size() < data->minTagsForValidObs) return;

  // sort detections by tagId
  std::sort(detections.begin(), detections.end(),
            AprilTags::TagDetection::sortByIdCompare);

  // check for duplicate tagIds (--> if found: wild Apriltags in image not
  // belonging to calibration target)
  // (only if we have more than 1 tag...)
  if (detections.size() > 1) {
    for (unsigned i = 0; i < detections.size() - 1; i++)
      if (detections[i].id == detections[i + 1].id) {
        std::cerr << "Wild Apriltag detected. Hide them!" << std::endl;
        return;
      }
  }

  // compute search radius for sub-pixel refinement depending on size of tag in
  // image
  std::vector<double> radiiRaw;
  for (unsigned i = 0; i < detections.size(); i++) {
    const double minimalRadius = 2.0;
    const double percentOfSideLength = 7.5;
    const double avgSideLength =
        static_cast<double>(detections[i].observedPerimeter) / 4.0;
    // use certain percentage of average side length as radius
    // subtract 1.0 since this radius is for displacement threshold; Search
    // region is slightly larger
    radiiRaw.emplace_back(std::max(
        minimalRadius, (percentOfSideLength / 100.0 * avgSideLength) - 1.0));
  }

  // convert corners to cv::Mat (4 consecutive corners form one tag)
  /// point ordering here
  ///          11-----10  15-----14
  ///          | TAG 2 |  | TAG 3 |
  ///          8-------9  12-----13
  ///          3-------2  7-------6
  ///    y     | TAG 0 |  | TAG 1 |
  ///   ^      0-------1  4-------5
  ///   |-->x
  cv::Mat tagCorners(4 * detections.size(), 2, CV_32F);

  for (unsigned i = 0; i < detections.size(); i++) {
    for (unsigned j = 0; j < 4; j++) {
      tagCorners.at<float>(4 * i + j, 0) = detections[i].p[j].first;
      tagCorners.at<float>(4 * i + j, 1) = detections[i].p[j].second;
    }
  }

  // store a copy of the corner list before subpix refinement
  cv::Mat tagCornersRaw = tagCorners.clone();

  // optional subpixel refinement on all tag corners (four corners each tag)
  if (data->doSubpixRefinement) {
    for (size_t i = 0; i < detections.size(); i++) {
      cv::Mat currentCorners(4, 2, CV_32F);
      for (unsigned j = 0; j < 4; j++) {
        currentCorners.at<float>(j, 0) = tagCorners.at<float>(4 * i + j, 0);
        currentCorners.at<float>(j, 1) = tagCorners.at<float>(4 * i + j, 1);
      }

      const int radius = static_cast<int>(std::ceil(radiiRaw[i] + 1.0));
      cv::cornerSubPix(
          image, currentCorners, cv::Size(radius, radius), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                           100, 0.01));

      for (unsigned j = 0; j < 4; j++) {
        tagCorners.at<float>(4 * i + j, 0) = currentCorners.at<float>(j, 0);
        tagCorners.at<float>(4 * i + j, 1) = currentCorners.at<float>(j, 1);
      }
    }
  }

  // insert the observed points into the correct location of the grid point
  // array
  /// point ordering
  ///          12-----13  14-----15
  ///          | TAG 2 |  | TAG 3 |
  ///          8-------9  10-----11
  ///          4-------5  6-------7
  ///    y     | TAG 0 |  | TAG 1 |
  ///   ^      0-------1  2-------3
  ///   |-->x

  for (unsigned int i = 0; i < detections.size(); i++) {
    // get the tag id
    int tagId = detections[i].id;

    // check maximum displacement from subpixel refinement
    const double radius = radiiRaw[i];
    const double tagMaxDispl2 = radius * radius;
    const double globalMaxDispl2 =
        data->maxSubpixDisplacement * data->maxSubpixDisplacement;
    const double subpixRefinementThreshold2 =
        globalMaxDispl2 > 0 ? std::min(globalMaxDispl2, tagMaxDispl2)
                            : tagMaxDispl2;

    // add four points per tag
    for (int j = 0; j < 4; j++) {
      int pointId = (tagId << 2) + j;

      // refined corners
      double corner_x = tagCorners.row(4 * i + j).at<float>(0);
      double corner_y = tagCorners.row(4 * i + j).at<float>(1);

      // raw corners
      double cornerRaw_x = tagCornersRaw.row(4 * i + j).at<float>(0);
      double cornerRaw_y = tagCornersRaw.row(4 * i + j).at<float>(1);

      // only add point if the displacement in the subpixel refinement is below
      // a given threshold
      double subpix_displacement_squarred =
          (corner_x - cornerRaw_x) * (corner_x - cornerRaw_x) +
          (corner_y - cornerRaw_y) * (corner_y - cornerRaw_y);

      // add all points, but only set active if the point has not moved to far
      // in the subpix refinement

      // TODO: We still get a few false positives here, e.g. when the whole
      // search region lies on an edge but the actual corner is not included.
      //       Maybe what we would need to do is actually checking a "corner
      //       score" vs "edge score" after refinement and discard all corners
      //       that are not more "cornery" than "edgy". Another possible issue
      //       might be corners, where (due to fisheye distortion), neighboring
      //       corners are in the search radius. For those we should check if in
      //       the radius there is really a clear single maximum in the corner
      //       score and otherwise discard the corner.

      if (subpix_displacement_squarred <= subpixRefinementThreshold2) {
        corners.emplace_back(corner_x, corner_y);
        ids.emplace_back(pointId);
        radii.emplace_back(radius);
      } else {
        corners_rejected.emplace_back(corner_x, corner_y);
        ids_rejected.emplace_back(pointId);
        radii_rejected.emplace_back(radius);
      }
    }
  }
}

}  // namespace basalt
