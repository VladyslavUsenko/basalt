cmake_minimum_required(VERSION 3.24)


add_executable(test_spline_opt src/test_spline_opt.cpp)
target_link_libraries(test_spline_opt GTest::gtest GTest::gtest_main basalt)

add_executable(test_vio src/test_vio.cpp)
target_link_libraries(test_vio GTest::gtest GTest::gtest_main basalt)

add_executable(test_nfr src/test_nfr.cpp)
target_link_libraries(test_nfr GTest::gtest GTest::gtest_main basalt)

add_executable(test_qr src/test_qr.cpp)
target_link_libraries(test_qr GTest::gtest GTest::gtest_main basalt)

add_executable(test_linearization src/test_linearization.cpp)
target_link_libraries(test_linearization GTest::gtest GTest::gtest_main basalt)

add_executable(test_patch src/test_patch.cpp)
target_link_libraries(test_patch GTest::gtest GTest::gtest_main basalt)

set(BASALT_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/basalt/utils)

target_include_directories(test_spline_opt PRIVATE ${BASALT_TEST_INCLUDE_DIRS})
target_include_directories(test_vio PRIVATE ${BASALT_TEST_INCLUDE_DIRS})
target_include_directories(test_nfr PRIVATE ${BASALT_TEST_INCLUDE_DIRS})
target_include_directories(test_qr PRIVATE ${BASALT_TEST_INCLUDE_DIRS})
target_include_directories(test_linearization PRIVATE ${BASALT_TEST_INCLUDE_DIRS})
target_include_directories(test_patch PRIVATE ${BASALT_TEST_INCLUDE_DIRS})

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
