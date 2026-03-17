cmake_minimum_required(VERSION 3.24)

project(basalt)

# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Didn't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BASALT_BUILD_SHARED_LIB "Build basalt core library as shared" OFF)
option(BASALT_FULLY_STATIC_BINARIES "Link executables fully static" ON)

if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(BASALT_MARCH_FLAGS "-march=${CXX_MARCH}")
else()
  message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Function for fully static linking (for non-GUI targets)
function(basalt_add_static_link_options target)
  if(BASALT_FULLY_STATIC_BINARIES AND UNIX AND NOT APPLE)
    target_link_options(${target} PRIVATE
      "$<$<CONFIG:Release>:-static>"
      "$<$<CONFIG:Release>:-static-libstdc++>"
      "$<$<CONFIG:Release>:-static-libgcc>"
    )
  endif()
endfunction()

# Function for GUI targets that need system libs (EGL, OpenGL, X11) dynamically
function(basalt_add_gui_link_options target)
  if(BASALT_FULLY_STATIC_BINARIES AND UNIX AND NOT APPLE)
    target_link_options(${target} PRIVATE
      "$<$<CONFIG:Release>:-static-libstdc++>"
      "$<$<CONFIG:Release>:-static-libgcc>"
    )
  endif()
endfunction()

set(CMAKE_REQUIRED_FLAGS "-Wno-error")

# save flags passed by user
set(BASALT_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_CIDEBUG  "-O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_CIRELWITHDEBINFO "-O3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")

set(BASALT_CXX_FLAGS "-Wall -Wextra -Werror -Wno-error=unused-parameter -ftemplate-backtrace-limit=0")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -Wno-error=deprecated-declarations")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=unused-variable -Wno-error=unused-but-set-variable")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=missing-template-arg-list-after-template-kw")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
     AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 20)
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -frelaxed-template-template-args")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-misleading-indentation -Wno-error=deprecated-copy")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=unused-but-set-variable")
  endif()

  if(NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
  endif()
else()
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=implicit-fallthrough")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=uninitialized")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=deprecated-copy")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-parentheses")
  endif()
endif()

if(APPLE)
  set(CMAKE_FIND_FRAMEWORK LAST)

  if(CMAKE_SYSTEM_VERSION VERSION_LESS 19.0.0)
    include_directories("/usr/local/opt/llvm/include/c++/v1")
    link_directories("/usr/local/opt/llvm/lib")
    add_compile_options("-nostdinc++")
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/opt/llvm/include/c++/v1")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected macOS with non-Apple clang")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Detected macOS with Apple clang")
    if(CMAKE_SYSTEM_VERSION VERSION_LESS 18.0.0)
      message(WARNING "Detected Darwin version ${CMAKE_SYSTEM_VERSION}, which is earlier than macos 10.14 Mojave.")
    endif()
  else()
    message(WARNING "Detected macOS with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()
elseif(UNIX)
  set(STD_CXX_FS stdc++fs)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected Linux with clang.")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Detected Linux with gcc.")
  else()
    message(WARNING "Detected Linux with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()
else()
  message(WARNING "Only Linux and macOS are currently supported")
endif()

set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

option(BASALT_INSTANTIATIONS_DOUBLE "Instantiate templates for Scalar=double." ON)
option(BASALT_INSTANTIATIONS_FLOAT "Instantiate templates for Scalar=float." ON)

if(BASALT_INSTANTIATIONS_DOUBLE)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_DOUBLE)
endif()
if(BASALT_INSTANTIATIONS_FLOAT)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_FLOAT)
endif()

set(CMAKE_CXX_FLAGS "${BASALT_CXX_FLAGS} ${BASALT_MARCH_FLAGS} ${BASALT_PASSED_CXX_FLAGS}")

find_package(Eigen3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc calib3d features2d imgcodecs)
find_package(Boost REQUIRED COMPONENTS date_time filesystem format program_options regex thread)
find_package(lz4 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(opengv CONFIG REQUIRED)
find_package(rosbag CONFIG REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
find_package(basalt-headers CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

get_target_property(BASALT_EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
if(NOT BASALT_EIGEN_INCLUDE_DIRS)
  message(FATAL_ERROR "Eigen3::Eigen does not publish include directories")
endif()

function(basalt_normalize_eigen_includes target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()

  get_target_property(_include_dirs ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _include_dirs)
    return()
  endif()

  set(_filtered_include_dirs "")
  set(_saw_eigen_include_dir OFF)
  foreach(_dir IN LISTS _include_dirs)
    if(_dir MATCHES ".*/eigen3?(/unsupported)?$")
      set(_saw_eigen_include_dir ON)
    else()
      list(APPEND _filtered_include_dirs "${_dir}")
    endif()
  endforeach()

  if(_saw_eigen_include_dir)
    list(APPEND _filtered_include_dirs ${BASALT_EIGEN_INCLUDE_DIRS})
    list(REMOVE_DUPLICATES _filtered_include_dirs)
    set_target_properties(${target_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_filtered_include_dirs}")
  endif()
endfunction()

foreach(_target opengv basalt::basalt-headers ${BASALT_PANGO_TARGETS})
  basalt_normalize_eigen_includes(${_target})
endforeach()

add_library(basalt::opencv INTERFACE IMPORTED)
set_property(TARGET basalt::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET basalt::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

# Build bundled apriltag library (needs basalt::opencv above)
add_subdirectory(thirdparty/apriltag EXCLUDE_FROM_ALL)

add_library(basalt::cli11 INTERFACE IMPORTED)
set_property(TARGET basalt::cli11 PROPERTY INTERFACE_LINK_LIBRARIES CLI11::CLI11)

add_library(basalt::magic_enum INTERFACE IMPORTED)
set_property(TARGET basalt::magic_enum PROPERTY INTERFACE_LINK_LIBRARIES magic_enum::magic_enum)

set(BASALT_PANGO_TARGETS ${Pangolin_LIBRARIES})

enable_testing()

# main library
if(BASALT_BUILD_SHARED_LIB)
  add_library(basalt SHARED)
else()
  add_library(basalt STATIC)
endif()

# Install basalt library
install(TARGETS basalt
  EXPORT BasaltTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# List all header and source files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
target_sources(basalt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/aprilgrid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/calibration_helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/cam_calib.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/cam_imu_calib.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/vignette.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/device/rs_t265.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/hash_bow/hash_bow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_euroc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_kitti.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_rosbag.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_uzh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/marg_data_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/block_diagonal.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/imu_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/landmark_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/landmark_block_abs_dynamic.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_abs_qr.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_abs_sc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_rel_sc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/frame_to_frame_optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/multiscale_frame_to_frame_optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patch_optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patterns.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/accumulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/linearize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/poses_linearize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/poses_optimize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/spline_linearize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/spline_optimize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/ba_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/cast_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/common_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/filesystem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/format.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/imu_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/keypoints.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/nfr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/sim_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/system_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/time_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/tracks.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/union_find.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vio_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vis_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/ba_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/landmark_database.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/marg_helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/nfr_mapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sc_ba_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_ba_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_keypoint_vio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_keypoint_vo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/vio_estimator.h
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/aprilgrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/calibraiton_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/vignette.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/dataset_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/marg_data_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/landmark_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_abs_qr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_abs_sc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_rel_sc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optical_flow/optical_flow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/keypoints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/system_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/time_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/vio_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/ba_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/landmark_database.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/marg_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/nfr_mapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sc_ba_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_ba_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_keypoint_vio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_keypoint_vo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/vio_estimator.cpp
)

target_link_libraries(basalt
  PUBLIC ${STD_CXX_FS} Eigen3::Eigen basalt::opencv basalt::basalt-headers TBB::tbb
  PRIVATE basalt::magic_enum rosbag apriltag opengv nlohmann_json::nlohmann_json fmt::fmt)
target_include_directories(basalt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ros/include)
target_compile_definitions(basalt PUBLIC ${BASALT_COMPILE_DEFINITIONS})
#target_compile_definitions(basalt PUBLIC BASALT_DISABLE_ASSERTS)

add_executable(basalt_calibrate src/calibrate.cpp src/calibration/cam_calib.cpp)
target_link_libraries(basalt_calibrate basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_calibrate)

add_executable(basalt_calibrate_imu src/calibrate_imu.cpp src/calibration/cam_imu_calib.cpp)
target_link_libraries(basalt_calibrate_imu basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_calibrate_imu)


add_executable(basalt_vio_sim src/vio_sim.cpp)
target_link_libraries(basalt_vio_sim basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_vio_sim)

add_executable(basalt_mapper_sim src/mapper_sim.cpp)
target_link_libraries(basalt_mapper_sim basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_mapper_sim)

# mapper sim native doesn't use template free interface
if(BASALT_INSTANTIATIONS_DOUBLE)
  add_executable(basalt_mapper_sim_naive src/mapper_sim_naive.cpp)
  target_link_libraries(basalt_mapper_sim_naive basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
  basalt_add_gui_link_options(basalt_mapper_sim_naive)
endif()

add_executable(basalt_mapper src/mapper.cpp)
target_link_libraries(basalt_mapper basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_mapper)


add_executable(basalt_opt_flow src/opt_flow.cpp)
target_link_libraries(basalt_opt_flow basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_opt_flow)

add_executable(basalt_vio src/vio.cpp)
target_link_libraries(basalt_vio basalt ${BASALT_PANGO_TARGETS} basalt::cli11 fmt::fmt)
basalt_add_gui_link_options(basalt_vio)

add_executable(basalt_time_alignment src/time_alignment.cpp)
target_link_libraries(basalt_time_alignment basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
basalt_add_gui_link_options(basalt_time_alignment)

add_executable(basalt_kitti_eval src/kitti_eval.cpp)
target_link_libraries(basalt_kitti_eval basalt::basalt-headers basalt::cli11)
basalt_add_static_link_options(basalt_kitti_eval)

find_package(realsense2 QUIET)
if(realsense2_FOUND AND NOT APPLE)
  add_executable(basalt_rs_t265_record src/rs_t265_record.cpp src/device/rs_t265.cpp)
  target_link_libraries(basalt_rs_t265_record basalt realsense2::realsense2 ${BASALT_PANGO_TARGETS} basalt::cli11)
  basalt_add_gui_link_options(basalt_rs_t265_record)

  add_executable(basalt_rs_t265_vio src/rs_t265_vio.cpp src/device/rs_t265.cpp)
  target_link_libraries(basalt_rs_t265_vio basalt realsense2::realsense2 ${BASALT_PANGO_TARGETS} basalt::cli11)
  basalt_add_gui_link_options(basalt_rs_t265_vio)

  # librealsense-file and rosbag both provide overlapping rosbag symbols.
  # Keep this relaxation local to the two T265 tools.
  if(UNIX AND NOT APPLE)
    target_link_options(basalt_rs_t265_record PRIVATE "LINKER:--allow-multiple-definition")
    target_link_options(basalt_rs_t265_vio PRIVATE "LINKER:--allow-multiple-definition")
  endif()
elseif(realsense2_FOUND AND APPLE)
  message(STATUS "Skipping T265 executables on macOS due to librealsense/rosbag static link conflicts.")
endif()



install(TARGETS basalt_calibrate basalt_calibrate_imu basalt_vio_sim basalt_mapper_sim basalt_mapper basalt_opt_flow basalt_vio basalt_kitti_eval basalt_time_alignment basalt
  EXPORT BasaltTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if(BASALT_INSTANTIATIONS_DOUBLE)
  install(TARGETS basalt_mapper_sim_naive
    EXPORT BasaltTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if(realsense2_FOUND AND NOT APPLE)
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


add_subdirectory(test)
