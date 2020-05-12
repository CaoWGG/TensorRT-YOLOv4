//
// Created by cao on 19-12-20.
//

#include "mish.h"
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)


__device__ float softplus_kernel(float x, const float threshold = 20) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1.);
}
__device__ half softplus_kernel(half x, const half threshold) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return hexp(x);    // too small
    return hlog(hexp(x) + half(1.));
}
__device__ half tanh_activate_kernel(half x){return (half(2.)/(half(1.) + hexp(half(-2.)*x)) - half(1.));}
__device__ float tanh_activate_kernel(float x){return (2./(1. + expf(-2.*x)) - 1.);}
template <typename T>
__global__ void mishKernel( int n, const T* input, T* output, const T MISH_THRESHOLD)
{

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        T x_val = input[idx];
        output[idx] = x_val * tanh_activate_kernel( softplus_kernel(x_val, MISH_THRESHOLD) );
    }
}

inline int computeMish(cudaStream_t stream, int n, const float* input, float* output)
{

    constexpr int blockSize = 1024;
    const int gridSize = (n + blockSize - 1) / blockSize;
    mishKernel<float><<<gridSize, blockSize, 0, stream>>>(n, input, output,20.);
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}
inline int computeMish(cudaStream_t stream, int n, const half* input, half* output)
{
    const int blockSize = 1024;
    const int gridSize = (n + blockSize - 1) / blockSize;
    mishKernel<half><<<gridSize, blockSize, 0, stream>>>(n, input, output,20.);
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

MishPlugin::MishPlugin():_initialized(false){


}
int MishPlugin::initialize() {
    if(_initialized) return 0;
    _initialized = true;
    return 0;
}
void MishPlugin::terminate() {
    if (!_initialized) {
        return;
    }
    _initialized = false;
}

MishPlugin::~MishPlugin() {
    terminate();
}

nvinfer1::Dims MishPlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) {
    assert(index == 0);
    assert(inputDims);
    assert(nbInputs == 1);
    return inputDims[0];
}
size_t MishPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int MishPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                         cudaStream_t stream) {
    nvinfer1::Dims input_dims = this->getInputDims(0);
    nvinfer1::DataType type = this->getDataType();
    const int C = input_dims.d[0];
    const int H = input_dims.d[1];
    const int W = input_dims.d[2];
    const int num = batchSize*C*H*W;
    switch (type)
    {
        case nvinfer1::DataType::kFLOAT:
        {
            const float* input_data = static_cast<const float*>(inputs[0]);
            float* out_data= static_cast<float*>(outputs[0]);
            computeMish(stream,num,input_data,out_data);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            const half* input_data = static_cast<const half*>(inputs[0]);
            half* out_data= static_cast<half*>(outputs[0]);
            computeMish(stream,num,input_data,out_data);
            break;
        }
        default: std::cerr << "error data type" << std::endl;;
    }
    return 0;
}
