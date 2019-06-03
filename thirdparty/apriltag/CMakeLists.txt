cmake_minimum_required(VERSION 3.2)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

#file(GLOB APRILTAG_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/apriltag-2016-12-01/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/apriltag-2016-12-01/common/*.c")
#include_directories(apriltag-2016-12-01)

include_directories(${OpenCV_INCLUDE_DIR})

file(GLOB APRILTAG_SRCS "ethz_apriltag2/src/*.cc")
include_directories(ethz_apriltag2/include)


include_directories(../../include)
include_directories(../basalt-headers/include)
include_directories(../basalt-headers/thirdparty/Sophus)

add_library(apriltag STATIC ${APRILTAG_SRCS} src/apriltag.cpp)

target_include_directories(apriltag PUBLIC include)
target_link_libraries(apriltag PUBLIC ${OpenCV_LIBS})


