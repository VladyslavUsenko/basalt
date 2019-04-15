cmake_minimum_required(VERSION 3.8)
project(basalt)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/basalt-headers/thirdparty/eigen")

string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWERCASE)
find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
if(DPKG_PROGRAM)
  execute_process(
    COMMAND ${DPKG_PROGRAM} --print-architecture
    OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif(DPKG_PROGRAM)


find_program(LSB_RELEASE_PROGRAM lsb_release DOC "lsb_release program of Debian-based systems")
if(LSB_RELEASE_PROGRAM)
execute_process(COMMAND ${LSB_RELEASE_PROGRAM} -rs
    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(${LSB_RELEASE_ID_SHORT} EQUAL "18.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.65.1, libboost-date-time1.65.1, libboost-program-options1.65.1, libboost-regex1.65.1, libopencv-dev, libglew2.0, libjpeg8, libpng16-16")
elseif(${LSB_RELEASE_ID_SHORT} EQUAL "16.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.58.0, libboost-date-time1.58.0, libboost-program-options1.58.0, libboost-regex1.58.0, libopencv-dev, libglew1.13, libjpeg8, libpng12-0")
endif(${LSB_RELEASE_ID_SHORT} EQUAL "18.04")

endif(LSB_RELEASE_PROGRAM)

string(TIMESTAMP PROJECT_VERSION_REVISION "%Y%m%d%H%M")

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Vladyslav Usenko <vlad.usenko@tum.de>")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "1")
SET(CPACK_PACKAGE_VERSION_PATCH "0-${PROJECT_VERSION_REVISION}~${LSB_RELEASE_ID_SHORT}")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS ${DEBIAN_DEPENDS})
SET(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME_LOWERCASE}_${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)


# Configure CCache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif(CCACHE_PROGRAM)

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release)
ENDIF()

IF(NOT CXX_MARCH)
   SET(CXX_MARCH native)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Flags used for CHECK_CXX_SOURCE_COMPILES
set(CMAKE_REQUIRED_FLAGS "-Wno-unused-variable -Wno-unused-value")


IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=${CXX_MARCH}")
   SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ftree-vectorize -march=${CXX_MARCH}")
   SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -Wno-deprecated-register -Qunused-arguments -fcolor-diagnostics -fopenmp")
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=${CXX_MARCH}")
   SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ftree-vectorize -march=${CXX_MARCH}")
   SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-deprecated-register -Wno-deprecated-declarations -Wno-sign-compare -Wno-exceptions -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-private-field -Qunused-arguments -fcolor-diagnostics -nostdinc++")
   include_directories(/usr/local/opt/llvm/include/c++/v1)
   link_directories(/usr/local/opt/llvm/lib)
   SET(STD_CXX_FS c++fs)
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=${CXX_MARCH}")
   SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ftree-vectorize -march=${CXX_MARCH}")
   SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -std=c++11 -Wno-misleading-indentation -Wno-sign-compare -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough -Wno-unused-parameter -Wno-deprecated-declarations -ftemplate-backtrace-limit=0 -fopenmp")
   SET(STD_CXX_FS stdc++fs)
ENDIF()

set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})

find_package(OpenCV REQUIRED core imgproc calib3d)
include_directories(${OpenCV_INCLUDE_DIR})
message(STATUS "Found OpenCV headers in: ${OpenCV_INCLUDE_DIR}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

add_subdirectory(thirdparty/ros)
add_subdirectory(thirdparty/apriltag)
add_subdirectory(thirdparty/DBoW3)


set(BUILD_SHARED_LIBS OFF CACHE BOOL "Enable BUILD_SHARED_LIBS")
set(BUILD_TESTS OFF CACHE BOOL "Enable BUILD_TESTS")

set(BUILD_PANGOLIN_LIBOPENEXR OFF CACHE BOOL "Enable BUILD_PANGOLIN_LIBOPENEXR")
set(BUILD_PANGOLIN_PYTHON OFF CACHE BOOL "Enable BUILD_PANGOLIN_PYTHON")
set(BUILD_EXAMPLES OFF CACHE BOOL "Enable BUILD_EXAMPLES")
add_subdirectory(thirdparty/opengv EXCLUDE_FROM_ALL)

# Hack to disable CPack in Pangolin.
macro(include)
  if(NOT ${ARGV0} STREQUAL "CPack")
    _include(${ARGN})
  endif()
endmacro()
add_subdirectory(thirdparty/Pangolin EXCLUDE_FROM_ALL)

include_directories(thirdparty/basalt-headers/thirdparty/Sophus)
include_directories(thirdparty/basalt-headers/thirdparty/cereal/include)
include_directories(thirdparty/basalt-headers/include)
include_directories(thirdparty/CLI11/include)
include_directories(thirdparty/fast/include)
include_directories(thirdparty/DBoW3/src/)

include_directories(include)


add_library(basalt SHARED
    src/io/dataset_io.cpp
    src/io/marg_data_io.cpp
    src/calibration/cam_calib.cpp
    src/calibration/cam_imu_calib.cpp
    src/calibration/calibraiton_helper.cpp
    src/calibration/vignette.cpp
    src/utils/vio_config.cpp
    src/optical_flow/optical_flow.cpp
    src/vi_estimator/keypoint_vio.cpp
    src/vi_estimator/keypoint_vio_linearize.cpp
    src/vi_estimator/vio_estimator.cpp
    src/vi_estimator/ba_base.cpp
    src/vi_estimator/nfr_mapper.cpp
    src/utils/keypoints.cpp)


target_link_libraries(basalt rosbag pangolin apriltag ${OPENGV_LIBS} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv ${STD_CXX_FS} DBoW3)


add_executable(basalt_calibrate src/calibrate.cpp)
target_link_libraries(basalt_calibrate ${Pangolin_LIBRARIES} apriltag ${OPENGV_LIBS} ${TBB_LIBRARIES} ${rosbag_LIBRARIES} basalt)

add_executable(basalt_calibrate_imu src/calibrate_imu.cpp)
target_link_libraries(basalt_calibrate_imu ${Pangolin_LIBRARIES} apriltag ${OPENGV_LIBS} ${TBB_LIBRARIES} ${rosbag_LIBRARIES} basalt)


add_executable(basalt_vio_sim src/vio_sim.cpp )
target_link_libraries(basalt_vio_sim ${Pangolin_LIBRARIES} opengv ${STD_CXX_FS} ${TBB_LIBRARIES} basalt)

add_executable(basalt_mapper_sim src/mapper_sim.cpp )
target_link_libraries(basalt_mapper_sim ${Pangolin_LIBRARIES} opengv ${TBB_LIBRARIES} basalt)

add_executable(basalt_mapper_sim_naive src/mapper_sim_naive.cpp)
target_link_libraries(basalt_mapper_sim_naive ${Pangolin_LIBRARIES} opengv ${TBB_LIBRARIES} basalt)

add_executable(basalt_mapper src/mapper.cpp)
target_link_libraries(basalt_mapper ${Pangolin_LIBRARIES} opengv ${STD_CXX_FS} ${TBB_LIBRARIES} basalt)


add_executable(basalt_opt_flow src/opt_flow.cpp)
target_link_libraries(basalt_opt_flow ${Pangolin_LIBRARIES} opengv ${STD_CXX_FS} basalt)

add_executable(basalt_vio src/vio.cpp)
target_link_libraries(basalt_vio ${Pangolin_LIBRARIES} opengv ${STD_CXX_FS} basalt)



install(TARGETS basalt_calibrate basalt_calibrate_imu basalt_vio_sim basalt_mapper_sim basalt_mapper_sim_naive basalt_mapper basalt_opt_flow basalt_vio basalt
    EXPORT BasaltTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

file(GLOB CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/*.json")
INSTALL(FILES ${CONFIG_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/data/orbvoc.dbow3
      DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/basalt
)


# Replace install() to do-nothing macro.
macro(install)
endmacro()
# Include subproject (or any other CMake code) with "disabled" install().
add_subdirectory(thirdparty/basalt-headers/test)
add_subdirectory(test)
# Restore original install() behavior.
macro(install)
    _install(${ARGN})
endmacro()



