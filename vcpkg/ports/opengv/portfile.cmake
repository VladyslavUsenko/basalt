vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://github.com/laurentkneip/opengv.git
    REF 91f4b19c73450833a40e463ad3648aae80b3a7f3
)

file(REMOVE_RECURSE "${SOURCE_PATH}/build")
vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt" "CXX_STANDARD 11" "CXX_STANDARD 17")
vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt" "set(CMAKE_BUILD_TYPE Release)" "if(NOT CMAKE_BUILD_TYPE)\n  set(CMAKE_BUILD_TYPE Release)\nendif()")
vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt" "ENDIF()\n\nset(CMAKE_MODULE_PATH" "ENDIF()\n\nadd_compile_options(-include cassert)\n\nset(CMAKE_MODULE_PATH")
vcpkg_replace_string("${SOURCE_PATH}/src/relative_pose/modules/main.cpp" " << samplingPoint.transpose()" "")

if(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    # Avoid OOM kills in arm64 CI builds caused by opengv's hardcoded -O3 flags.
    vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt" "add_definitions (\n    -O3" "add_definitions (\n    -O0")
endif()

# opengv ships an old FindEigen module that parses removed Eigen version macros.
# Replace it with a shim using Eigen3 CONFIG from vcpkg.
file(WRITE "${SOURCE_PATH}/modules/FindEigen.cmake" [=[
find_package(Eigen3 CONFIG REQUIRED)
set(EIGEN_FOUND TRUE)
if(NOT EIGEN3_INCLUDE_DIR)
  get_target_property(_eigen3_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  if(_eigen3_inc)
    list(GET _eigen3_inc 0 EIGEN3_INCLUDE_DIR)
  endif()
endif()
set(EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
set(EIGEN_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
set(EIGEN_VERSION "3.4.0")
]=])

set(_opengv_cmake_options
    -DBUILD_TESTS=OFF
    -DBUILD_PYTHON=OFF
    -DCMAKE_DEBUG_POSTFIX=
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
)
if(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    list(APPEND _opengv_cmake_options
        "-DCMAKE_C_FLAGS_DEBUG=-O0 -g0"
        "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g0"
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${_opengv_cmake_options}
)

set(_opengv_install_args)
if(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    list(APPEND _opengv_install_args DISABLE_PARALLEL)
endif()
vcpkg_cmake_install(${_opengv_install_args})

foreach(_opengv_targets
    "${CURRENT_PACKAGES_DIR}/lib/cmake/opengv-1.0/opengvTargets.cmake"
    "${CURRENT_PACKAGES_DIR}/debug/lib/cmake/opengv-1.0/opengvTargets.cmake")
  if(EXISTS "${_opengv_targets}")
    file(READ "${_opengv_targets}" _opengv_targets_content)
    string(REGEX REPLACE
      "[^;]+/vcpkg_installed/[^/]+/include/eigen3;[^;]+/vcpkg_installed/[^/]+/include/eigen3/unsupported"
      "\${_IMPORT_PREFIX}/include/eigen3;\${_IMPORT_PREFIX}/include/eigen3/unsupported"
      _opengv_targets_content
      "${_opengv_targets_content}")
    string(REGEX REPLACE
      "[^;]+/vcpkg_installed/[^/]+/debug/include"
      "\${_IMPORT_PREFIX}/include"
      _opengv_targets_content
      "${_opengv_targets_content}")
    string(REPLACE
      "\${_IMPORT_PREFIX}/debug/include"
      "\${_IMPORT_PREFIX}/include"
      _opengv_targets_content
      "${_opengv_targets_content}")
    file(WRITE "${_opengv_targets}" "${_opengv_targets_content}")
  endif()
endforeach()

if(EXISTS "${CURRENT_PACKAGES_DIR}/lib/cmake/opengv-1.0")
    vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/opengv-1.0 PACKAGE_NAME opengv)
elseif(EXISTS "${CURRENT_PACKAGES_DIR}/share/opengv/opengvConfig.cmake")
    vcpkg_cmake_config_fixup(CONFIG_PATH share/opengv PACKAGE_NAME opengv)
endif()

vcpkg_copy_pdbs()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

file(INSTALL "${CURRENT_PORT_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.txt")
