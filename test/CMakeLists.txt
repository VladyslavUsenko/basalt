cmake_minimum_required(VERSION 3.2)

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})

set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)


include_directories(../thirdparty/basalt-headers/test/include)


add_executable(test_image src/test_image.cpp)
target_link_libraries(test_image ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${OpenCV_LIBS} basalt)

add_executable(test_spline_opt src/test_spline_opt.cpp)
target_link_libraries(test_spline_opt ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${TBB_LIBRARIES} opengv basalt)

add_executable(test_vio src/test_vio.cpp)
target_link_libraries(test_vio ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${TBB_LIBRARIES} opengv basalt)

add_executable(test_nfr src/test_nfr.cpp)
target_link_libraries(test_nfr ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${TBB_LIBRARIES} opengv basalt)


enable_testing()

add_test(test_image test_image COMMAND Test)
add_test(test_spline_opt test_spline_opt COMMAND Test)
add_test(test_vio test_vio COMMAND Test)
add_test(test_nfr test_nfr COMMAND Test)
