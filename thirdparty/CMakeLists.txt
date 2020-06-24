cmake_minimum_required(VERSION 3.10)

add_subdirectory(ros EXCLUDE_FROM_ALL)
add_subdirectory(apriltag EXCLUDE_FROM_ALL)


set(BUILD_SHARED_LIBS OFF CACHE BOOL "Enable BUILD_SHARED_LIBS")
set(BUILD_TESTS OFF CACHE BOOL "Enable BUILD_TESTS")
set(BUILD_TOOLS OFF CACHE BOOL "Enable BUILD_TOOLS")
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)

set(BUILD_PANGOLIN_LIBOPENEXR OFF CACHE BOOL "Enable BUILD_PANGOLIN_LIBOPENEXR")
set(BUILD_PANGOLIN_PYTHON OFF CACHE BOOL "Enable BUILD_PANGOLIN_PYTHON")
set(BUILD_EXAMPLES OFF CACHE BOOL "Enable BUILD_EXAMPLES")
set(BUILD_PANGOLIN_LIBREALSENSE OFF CACHE BOOL "Enable librealsense")
set(BUILD_PANGOLIN_LIBREALSENSE2 OFF CACHE BOOL "Enable librealsense2")

set(EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
set(EIGEN_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")


# Hack to disable march=native in opengv
macro(add_definitions)
  if(NOT ${ARGV0} STREQUAL "-march=native")
    _add_definitions(${ARGN})
  endif()
endmacro()
add_subdirectory(opengv EXCLUDE_FROM_ALL)

# Hack to disable CPack in Pangolin.
macro(include)
  if(NOT ${ARGV0} STREQUAL "CPack")
    _include(${ARGN})
  endif()
endmacro()
add_subdirectory(Pangolin EXCLUDE_FROM_ALL)


# fix aprilgrid
target_compile_options(apriltag PRIVATE "-Wno-unused-private-field")

# fix opengv: c++17 and debug postfix
set_target_properties(opengv PROPERTIES
  CXX_STANDARD 17
  DEBUG_POSTFIX "")

# fix opengv: compile options (CMAKE_CXX_FLAGS is overwritten by Opengv)
target_compile_options(opengv PRIVATE
  -Wno-unused-private-field
  -march=${CXX_MARCH}
  ${BASALT_PASSED_CXX_FLAGS})

# TODO: enable once mpark issue is fixed upstream pangolin
#set_target_properties(pangolin PROPERTIES
#  CXX_STANDARD 17)

# fix pangolin: gcc
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(pangolin PRIVATE "-Wno-implicit-fallthrough")
endif()

# fix pangolin: macOS
if(APPLE)
  target_compile_options(pangolin PRIVATE "-Wno-objc-missing-super-calls")
endif()

# fix pangolin: macOS >= 10.14 Mojave
if(APPLE AND CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL 18.0.0)
  target_compile_options(pangolin PRIVATE "-Wno-deprecated-declarations")
endif()

# fix pangolin: clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(pangolin PRIVATE "-Wno-null-pointer-arithmetic")
endif()

# fix pangolin: clang >= 8.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0.0)
  target_compile_options(pangolin PRIVATE "-Wno-defaulted-function-deleted")
endif()

# fix pangolin: clang >= 10.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0.0)
  target_compile_options(pangolin PRIVATE "-Wno-deprecated-copy")
endif()

# fix pangolin: GCC >= 9.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
  target_compile_options(pangolin PRIVATE "-Wno-stringop-truncation;-Wno-deprecated-copy")
endif()

# check here the directories for the pangolin and opengv targets, and
# confirm that the eigen-related include dirs match.
foreach(_target opengv pangolin)
  get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_dir IN LISTS _include_dirs)
    if("${_dir}" MATCHES ".*/eigen3?(/unsupported)?$")
      if(NOT _dir MATCHES "^${EIGEN3_INCLUDE_DIR}.*")
      endif()
    endif()
  endforeach()
endforeach()

# opengv and pangolin assume that eigen is found outside the source
# directory and thus include it in INTERFACE_INCLUDE_DIRECTORIES,
# which makes cmake complain that that property contains paths in the
# source folder. Thus, we update the property to only include these
# eigen paths in the "BUILD_INTERFACE" (not "INSTALL").
if (EIGEN3_INCLUDE_DIR MATCHES "^${CMAKE_SOURCE_DIR}.*")
  foreach(_target opengv pangolin)
    get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    set(_include_dirs_new "")
    foreach(_dir IN LISTS _include_dirs)
      if(_dir MATCHES ".*/eigen(/unsupported)?$")
        string(REGEX REPLACE "(^${CMAKE_SOURCE_DIR}.*$)" "$<BUILD_INTERFACE:\\1>" _dir "${_dir}")
      endif()
      list(APPEND _include_dirs_new "${_dir}")
    endforeach()
    set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs_new}")
  endforeach()
endif()



