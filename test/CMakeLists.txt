cmake_minimum_required(VERSION 3.10...3.18)

# Note: add_subdirectory(googletest ...) is called in basalt-headers

include_directories(../thirdparty/basalt-headers/test/include)


add_executable(test_spline_opt src/test_spline_opt.cpp)
target_link_libraries(test_spline_opt gtest gtest_main basalt)

add_executable(test_vio src/test_vio.cpp)
target_link_libraries(test_vio gtest gtest_main basalt)

add_executable(test_nfr src/test_nfr.cpp)
target_link_libraries(test_nfr gtest gtest_main basalt)

add_executable(test_qr src/test_qr.cpp)
target_link_libraries(test_qr gtest gtest_main basalt)

add_executable(test_linearization src/test_linearization.cpp)
target_link_libraries(test_linearization gtest gtest_main basalt)

add_executable(test_patch src/test_patch.cpp)
target_link_libraries(test_patch gtest gtest_main basalt)

enable_testing()

include(GoogleTest)

#gtest_discover_tests(test_spline_opt DISCOVERY_TIMEOUT 60)
#gtest_discover_tests(test_vio DISCOVERY_TIMEOUT 60)
#gtest_discover_tests(test_nfr DISCOVERY_TIMEOUT 60)

gtest_add_tests(TARGET test_spline_opt AUTO)
gtest_add_tests(TARGET test_vio AUTO)
gtest_add_tests(TARGET test_nfr AUTO)
gtest_add_tests(TARGET test_qr AUTO)
gtest_add_tests(TARGET test_linearization AUTO)
gtest_add_tests(TARGET test_patch AUTO)
