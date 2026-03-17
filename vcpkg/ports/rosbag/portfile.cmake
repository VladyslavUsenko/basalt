vcpkg_from_git(
    OUT_SOURCE_PATH ROS_COMM_SOURCE
    URL https://github.com/ros/ros_comm.git
    REF 194c776737134a429619f6851f1850734843e210
)

vcpkg_from_git(
    OUT_SOURCE_PATH ROSCPP_CORE_SOURCE
    URL https://github.com/NikolausDemmel/roscpp_core.git
    REF 1a74781ce707b1a2198d1bd6799b892b3c1fa195
)

set(PORT_SOURCE "${CURRENT_BUILDTREES_DIR}/src/rosbag-src")
file(MAKE_DIRECTORY "${PORT_SOURCE}")

configure_file("${CURRENT_PORT_DIR}/cmake/CMakeLists.txt.in" "${PORT_SOURCE}/CMakeLists.txt" @ONLY)
configure_file("${CURRENT_PORT_DIR}/cmake/rosbagConfig.cmake.in" "${PORT_SOURCE}/rosbagConfig.cmake.in" COPYONLY)

vcpkg_cmake_configure(
    SOURCE_PATH "${PORT_SOURCE}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH share/rosbag)

vcpkg_copy_pdbs()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

file(INSTALL "${CURRENT_PORT_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright" "ROS sources (BSD-style licenses) bundled for Basalt compatibility.\n")
