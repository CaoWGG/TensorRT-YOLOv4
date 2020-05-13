#include "darknetadd.h"
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

template <typename Data>
__global__ void shortcut_kernel(int size, int src_outputs, int add_outputs,const Data *in,const Data *add, Data *out)
{
    const int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int src_id = id;
    const int src_i = src_id % src_outputs;
    src_id /= src_outputs;
    int src_b = src_id;

    Data out_val = in[id];

    if (src_i < add_outputs) {
        int add_index = add_outputs*src_b + src_i;
        out_val += add[add_index];
    }
    out[id] = out_val;
}
template <typename Data>
inline int addlayer(cudaStream_t stream, int n,int input1size,int input2size, const Data* input1,const Data* input2,Data * output)
{
    const int blockSize = 1024;
    const int gridSize = (n + blockSize - 1) / blockSize;
    shortcut_kernel<<<gridSize, blockSize, 0, stream>>>(n, input1size,input2size,input1,input2,output);
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}
ADDPlugin::ADDPlugin():_initialized(false){


}
int ADDPlugin::initialize() {
    if(_initialized) return 0;
    _initialized = true;
    return 0;
}
void ADDPlugin::terminate() {
    if (!_initialized) {
        return;
    }
    _initialized = false;
}

ADDPlugin::~ADDPlugin() {
    terminate();
}

nvinfer1::Dims ADDPlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) {
    assert(index == 0);
    assert(inputDims);
    assert(nbInputs == 2);
    return inputDims[0];
}
size_t ADDPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int ADDPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                        cudaStream_t stream) {
    nvinfer1::Dims input_dims1 = this->getInputDims(0);
    nvinfer1::Dims input_dims2 = this->getInputDims(1);
    nvinfer1::DataType type = this->getDataType();
    const int size_1 = input_dims1.d[0]*input_dims1.d[1]*input_dims1.d[2];
    const int size_2 = input_dims2.d[0]*input_dims2.d[1]*input_dims2.d[2];
    const int num = batchSize*size_1;
    switch (type)
    {
        case nvinfer1::DataType::kFLOAT:
        {
            const float* input_data_1 = static_cast<const float*>(inputs[0]);
            const float* input_data_2 = static_cast<const float*>(inputs[1]);
            float* out_data= static_cast<float*>(outputs[0]);
            addlayer(stream,num,size_1,size_2,input_data_1,input_data_2,out_data);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            const half* input_data_1 = static_cast<const half*>(inputs[0]);
            const half* input_data_2 = static_cast<const half*>(inputs[1]);
            half* out_data= static_cast<half*>(outputs[0]);
            addlayer(stream,num,size_1,size_2,input_data_1,input_data_2,out_data);
            break;
        }
        default: std::cerr << "error data type" << std::endl;;
    }
    return 0;
}
