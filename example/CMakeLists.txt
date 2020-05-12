cmake_minimum_required(VERSION 3.5)
project(example LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -std=c++11")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
find_package(OpenCV REQUIRED)
include_directories(
        ../include
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS})

message(STATUS ${OpenCV_LIBS})
add_executable(buildEngine  buildEngine.cpp )
target_link_libraries(buildEngine trt ${OpenCV_LIBS})

add_executable(runDet runDet.cpp )
target_link_libraries(runDet trt ${OpenCV_LIBS})