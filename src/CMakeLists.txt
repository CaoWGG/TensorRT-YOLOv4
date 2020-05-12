cmake_minimum_required(VERSION 3.5)
project(trt LANGUAGES CXX )
find_package(CUDA REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
find_library(TENSORRT_INFER libnvinfer.so
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_INFER_PLUGIN libnvinfer_plugin.so
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
include_directories(
        ./
        ../include
        ../onnx-tensorrt
        ${TENSORRT_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast ")
set(CUDA_NVCC_FLAGS  "-D_FORCE_INLINES -Xcompiler -fPIC -gencode arch=compute_${GPU_ARCHS},code=sm_${GPU_ARCHS} -gencode arch=compute_${GPU_ARCHS},code=compute_${GPU_ARCHS}")
set(SRC_LIST resize.cu trt.cpp)
cuda_add_library(trt SHARED ${SRC_LIST})
target_link_libraries(trt ${TENSORRT_INFER_PLUGIN} ${TENSORRT_INFER} mynvonnxparser mynvonnxparser_runtime)