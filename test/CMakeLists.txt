cmake_minimum_required(VERSION 3.10)

# Note: add_subdirectory(googletest ...) is called in basalt-headers

include_directories(../thirdparty/basalt-headers/test/include)


add_executable(test_spline_opt src/test_spline_opt.cpp)
target_link_libraries(test_spline_opt gtest gtest_main basalt)

add_executable(test_vio src/test_vio.cpp)
target_link_libraries(test_vio gtest gtest_main basalt)

add_executable(test_nfr src/test_nfr.cpp)
target_link_libraries(test_nfr gtest gtest_main basalt)


enable_testing()

include(GoogleTest)

#gtest_discover_tests(test_spline_opt DISCOVERY_TIMEOUT 60)
#gtest_discover_tests(test_vio DISCOVERY_TIMEOUT 60)
#gtest_discover_tests(test_nfr DISCOVERY_TIMEOUT 60)

gtest_add_tests(TARGET test_spline_opt AUTO)
gtest_add_tests(TARGET test_vio AUTO)
gtest_add_tests(TARGET test_nfr AUTO)
