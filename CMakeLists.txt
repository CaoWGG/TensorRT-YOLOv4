cmake_minimum_required(VERSION 3.5)
project(TensorRT-YOLOv4)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Relaese)
set(CMAKE_CXX_STANDARD  11)
set(GPU_ARCHS 61)
set(TENSORRT_ROOT /usr/local/TensorRT-5.0.2.6)
add_subdirectory(onnx-tensorrt)
add_subdirectory(src)
add_subdirectory(example)