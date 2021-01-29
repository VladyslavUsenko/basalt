cmake_minimum_required(VERSION 3.10)

include("thirdparty/basalt-headers/cmake_modules/PreProjectWorkarounds.cmake")

project(basalt)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/basalt-headers/thirdparty/eigen")
endif()

string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWERCASE)
find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
if(DPKG_PROGRAM)
  execute_process(
    COMMAND ${DPKG_PROGRAM} --print-architecture
    OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(DPKG_PROGRAM)


find_program(LSB_RELEASE_PROGRAM lsb_release DOC "lsb_release program of Debian-based systems")
if(LSB_RELEASE_PROGRAM)
  execute_process(COMMAND ${LSB_RELEASE_PROGRAM} -rs
    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(${LSB_RELEASE_ID_SHORT} EQUAL "20.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.71.0, libboost-date-time1.71.0, libboost-program-options1.71.0, libboost-regex1.71.0, libopencv-dev, libglew2.1, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "18.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.65.1, libboost-date-time1.65.1, libboost-program-options1.65.1, libboost-regex1.65.1, libopencv-dev, libglew2.0, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "16.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.58.0, libboost-date-time1.58.0, libboost-program-options1.58.0, libboost-regex1.58.0, libopencv-dev, libglew1.13, libjpeg8, libpng12-0, libstdc++6, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")
  endif()

endif(LSB_RELEASE_PROGRAM)

string(TIMESTAMP PROJECT_VERSION_REVISION "%Y%m%d%H%M")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Vladyslav Usenko <vlad.usenko@tum.de>")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0-${PROJECT_VERSION_REVISION}~${LSB_RELEASE_ID_SHORT}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS ${DEBIAN_DEPENDS})
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME_LOWERCASE}_${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)


# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release)
endif()

if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Flags used for CHECK_CXX_SOURCE_COMPILES
set(CMAKE_REQUIRED_FLAGS "-Wno-error")


# save flags passed by user
set(BASALT_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                      # cmake default: "-O3 -DNDEBUG"
set(CMAKE_CXX_FLAGS_CIDEBUG  "-O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # CI version with no debug symbols
set(CMAKE_CXX_FLAGS_CIRELWITHDEBINFO "-O3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # CI version with no debug symbols

# base set of compile flags
set(BASALT_CXX_FLAGS "-Wall -Wextra -Werror -Wno-error=unused-parameter -ftemplate-backtrace-limit=0")

# clang-specific compile flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -frelaxed-template-template-args -Wno-error=deprecated-declarations")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 10
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-misleading-indentation -Wno-error=deprecated-copy")
  endif()

  #   - Added TBB_USE_GLIBCXX_VERSION macro to specify the version of GNU
  #     libstdc++ when it cannot be properly recognized, e.g. when used
  #     with Clang on Linux* OS. Adopted from https://github.com/wjakob/tbb
  if(NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
  endif()
else()
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=implicit-fallthrough")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
      # These are disabled to avoid lot's of warnings in Eigen code with gcc-9
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=deprecated-copy")
  endif()
endif()


# Set platform / compiler specific compile flags and checks
if(APPLE)
  # Need to investigate how to reliably detect and use OpenMP on macOS...
#  set(USE_OPENMP_DEFAULT OFF)

  # Among others, setting CMAKE_FIND_FRAMEWORK to LAST fixed issues
  # with installed Mono that contains old headers (libpng, ...).
  # See: https://github.com/openMVG/openMVG/issues/1349#issuecomment-401492811
  set(CMAKE_FIND_FRAMEWORK LAST)

  if(CMAKE_SYSTEM_VERSION VERSION_LESS 19.0.0)
      # use brewed llvm's libc++
      include_directories("/usr/local/opt/llvm/include/c++/v1")
      link_directories("/usr/local/opt/llvm/lib")
      add_compile_options("-nostdinc++")
      #set(STD_CXX_FS c++fs)

      # Workaround for cmake not to filter the manually added standard include path
      # See: https://gitlab.kitware.com/cmake/cmake/issues/19227#note_669894
      list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/opt/llvm/include/c++/v1")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected macOS with non-Apple clang")

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Detected macOS with Apple clang")
    # Apple clang on macOS < 10.14 Mojave is too old
    if(CMAKE_SYSTEM_VERSION VERSION_LESS 18.0.0)
      message(WARNING "Detected Darwin version ${CMAKE_SYSTEM_VERSION}, which is earlier than macos 10.14 Mojave. Apple clang is too old and not supported. Use clang from homebrew.")
    endif()

  else()
    message(WARNING "Detected macOS with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

elseif(UNIX)
#  set(USE_OPENMP_DEFAULT ON)

  # assume libstdc++
  set(STD_CXX_FS stdc++fs)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(LINUX_CLANG 1)
    message(STATUS "Detected Linux with clang.")
    message(WARNING "Clang on Linux is currently not fully supported. You'll likely need to get a recent version of TBB.")

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Detected Linux with gcc.")

  else()
    message(WARNING "Detected Linux with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

else()
  message(WARNING "Only Linux and macOS are currently supported")
endif()


# OpenMP option and compile flags
#
# Note: OpenMP and TBB don't mix well, so we disable Eigen's parallelization.
# It's trying to paralellize matrix products during SC, which we run in a parallel_reduce using TBB.
# Turns out using OpenMP can slow down the computby factor 10-100x! So for now we discable it completely.
# One place where Eigen's parallelization could still have been useful is the CG solver in the mapper.
# We could in the future investiagte other implementations (paralellized with TBB) or selectively enabling
# Eigen's parallelization just for CG, setting number of threads to 1 everywhere else.
# Another way to ensure Eigen doesn't use OpenMP regardless of how it was built is setting the environment
# variable OMP_NUM_THREADS=1 beofre running the application.
#
# See: https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
#
# If we enable BLAS / LAPACK either directly or via thirdparty libs like ceres,
# make sure to disable OpenMP for the linked BLAS library. In particular on Ubuntu it seems OpenBLAS is often installed,
# and it can have similar issues in multithreaded applications if it's own parallelization with OpenMP is enabled.
# You can set the environment varaibles OPENBLAS_NUM_THREADS=1 or OMP_NUM_THREADS=1. This is also mentioned in the ceres
# installation documentation.
#
# See also: https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
#
# Set EIGEN_DONT_PARALLELIZE to be sure it doesn't use openmp,
# just in case some dependency enables openmp without us noticing.
set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

#option(USE_OPENMP "Use OpenMP (e.g. for parallel computation in Eigen)" ${USE_OPENMP_DEFAULT})
#if(USE_OPENMP)
#  message(STATUS "OpenMP Enabled")
#  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -fopenmp")
#else()
#  message(STATUS "OpenMP Disabled")
#endif()


# setup combined compiler flags
set(CMAKE_CXX_FLAGS "${BASALT_CXX_FLAGS} -march=${CXX_MARCH} ${BASALT_PASSED_CXX_FLAGS}")


set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 EXACT REQUIRED MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR MATCHES "^${EIGEN_ROOT}")
  message(WARNING "Found Eigen headers are outside of specified EIGEN_ROOT '${EIGEN_ROOT}'")
endif()

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})
message(STATUS "TBB_INTERFACE_VERSION: ${TBB_INTERFACE_VERSION}")
if (TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()

find_package(OpenCV REQUIRED core imgproc calib3d highgui)
include_directories(${OpenCV_INCLUDE_DIR})
message(STATUS "Found OpenCV headers in: ${OpenCV_INCLUDE_DIR}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")


add_subdirectory(thirdparty)


include_directories(thirdparty/basalt-headers/thirdparty/Sophus)
include_directories(thirdparty/basalt-headers/thirdparty/cereal/include)
include_directories(thirdparty/basalt-headers/include)
include_directories(thirdparty/CLI11/include)
include_directories(thirdparty/fast/include)

include_directories(include)


add_library(basalt SHARED
  src/io/dataset_io.cpp
  src/io/marg_data_io.cpp
  src/calibration/aprilgrid.cpp
  src/calibration/calibraiton_helper.cpp
  src/calibration/vignette.cpp
  src/utils/vio_config.cpp
  src/optical_flow/optical_flow.cpp
  src/vi_estimator/keypoint_vio.cpp
  src/vi_estimator/keypoint_vio_linearize.cpp
  src/vi_estimator/keypoint_vo.cpp
  src/vi_estimator/vio_estimator.cpp
  src/vi_estimator/ba_base.cpp
  src/vi_estimator/nfr_mapper.cpp
  src/vi_estimator/landmark_database.cpp
  src/utils/keypoints.cpp)


target_link_libraries(basalt PUBLIC ${TBB_LIBRARIES} ${STD_CXX_FS} ${OpenCV_LIBS} PRIVATE rosbag apriltag opengv)


add_executable(basalt_calibrate src/calibrate.cpp src/calibration/cam_calib.cpp)
target_link_libraries(basalt_calibrate basalt pangolin)

add_executable(basalt_calibrate_imu src/calibrate_imu.cpp src/calibration/cam_imu_calib.cpp)
target_link_libraries(basalt_calibrate_imu basalt pangolin)


add_executable(basalt_vio_sim src/vio_sim.cpp)
target_link_libraries(basalt_vio_sim basalt pangolin)

add_executable(basalt_mapper_sim src/mapper_sim.cpp)
target_link_libraries(basalt_mapper_sim basalt pangolin)

add_executable(basalt_mapper_sim_naive src/mapper_sim_naive.cpp)
target_link_libraries(basalt_mapper_sim_naive basalt pangolin)

add_executable(basalt_mapper src/mapper.cpp)
target_link_libraries(basalt_mapper basalt pangolin)


add_executable(basalt_opt_flow src/opt_flow.cpp)
target_link_libraries(basalt_opt_flow basalt pangolin)

add_executable(basalt_vio src/vio.cpp)
target_link_libraries(basalt_vio basalt pangolin)

add_executable(basalt_time_alignment src/time_alignment.cpp)
target_link_libraries(basalt_time_alignment basalt pangolin)

add_executable(basalt_kitti_eval src/kitti_eval.cpp)
target_link_libraries(basalt_kitti_eval)

find_package(realsense2 QUIET)
if(realsense2_FOUND)
  add_executable(basalt_rs_t265_record src/rs_t265_record.cpp src/device/rs_t265.cpp)
  target_link_libraries(basalt_rs_t265_record basalt realsense2::realsense2 ${OpenCV_LIBS} pangolin)

  add_executable(basalt_rs_t265_vio src/rs_t265_vio.cpp src/device/rs_t265.cpp)
  target_link_libraries(basalt_rs_t265_vio basalt realsense2::realsense2 pangolin)
endif()



install(TARGETS basalt_calibrate basalt_calibrate_imu basalt_vio_sim basalt_mapper_sim basalt_mapper_sim_naive basalt_mapper basalt_opt_flow basalt_vio basalt_kitti_eval basalt_time_alignment basalt
  EXPORT BasaltTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if(realsense2_FOUND)
    install(TARGETS basalt_rs_t265_record basalt_rs_t265_vio
      EXPORT BasaltTargets
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
      LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

file(GLOB SCRIPTS_TO_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/scripts/basalt_*.py")
install(PROGRAMS ${SCRIPTS_TO_INSTALL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

file(GLOB CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/*.json")
install(FILES ${CONFIG_FILES}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/basalt)


# Replace install() to do-nothing macro.
macro(install)
endmacro()
# Include subproject (or any other CMake code) with "disabled" install().
enable_testing()
add_subdirectory(thirdparty/basalt-headers/test)
add_subdirectory(test)
# Restore original install() behavior.
macro(install)
  _install(${ARGN})
endmacro()
