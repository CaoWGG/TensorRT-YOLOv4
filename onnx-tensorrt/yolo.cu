#include "yolo.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)


__device__ float sigmoid(float data){ return 1./(1. + expf(-data)); };
__global__ void yoloKernel(const int n,const float * input, float* output, const int* anchors,int anchor_num,
                           int classes,int height,int width,float down_stride,float thresh){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    extern __shared__ int shared_anchors[];
    if(threadIdx.x < anchor_num*2){
        shared_anchors[threadIdx.x] = anchors[threadIdx.x];
    }
    __syncthreads();
    int row = idx % width;
    int col = (idx / width) % height;
    int anchor_id = (idx / width / height)% anchor_num;
    int batch_id = idx/width/height/anchor_num;
    int C = anchor_num*(classes+5);
    int stride = width*height;
    int begin_id =  ((batch_id * C + anchor_id*(classes + 5))*height+col)*width+row;
    float conf_prob =sigmoid(input[begin_id + 4*stride]);
    if(conf_prob > thresh) {
        int class_id = -1;
        float max_prob = thresh;
        for (int c = 0;c<classes;++c){
            int cls_id = begin_id + stride*(c + 5);
            float cls_prob =  sigmoid(input[cls_id]) *conf_prob ;
            if(cls_prob > max_prob){
                max_prob = cls_prob;
                class_id = c;
            }
        }
        if(class_id >= 0){
            int resCount = (int)atomicAdd(output,1);
            float * data = output + 1 + resCount*7;
            // x1,y1,x2,y2,cls,conf,batch_id
            data[0] = (row + sigmoid(input[begin_id]))*down_stride;
            data[1] = (col  + sigmoid(input[begin_id+stride]))*down_stride;
            data[2] = expf(input[begin_id+2*stride]) * (float)shared_anchors[2*anchor_id];
            data[3] = expf(input[begin_id+3*stride]) * (float)shared_anchors[2*anchor_id + 1];
            data[4] = class_id;
            data[5] = max_prob;
            data[6] = batch_id;
        }
    }
}
inline int yololayer(cudaStream_t stream, int n, const float* input, float* output,
                     const int* anchors,int anchor_num,int classes,int height,int width,float down_stride,float thresh)
{

    constexpr int blockSize = 1024;
    const int gridSize = (n + blockSize - 1) / blockSize;
    yoloKernel<<<gridSize, blockSize, anchor_num*2, stream>>>(n, input, output,anchors,anchor_num,classes,height,width,down_stride,thresh);
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

YOLOPlugin::YOLOPlugin(const std::vector<int>& anchors, const int anchorNum,
                        const int classes, const int downStride, const float inferThresh):
                        _initialized(false),anchors(anchors),anchorNum(anchorNum),
                        classes(classes),downStride(downStride),inferThresh(inferThresh){


}
bool YOLOPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const {

    return (type == nvinfer1::DataType::kFLOAT && format==nvinfer1::PluginFormat::kNCHW);
}
int YOLOPlugin::initialize() {
    if(_initialized) return 0;
    CHECK_CUDA(cudaMalloc((void**)&cudaAnchors,anchorNum*2* sizeof(int)));
    CHECK_CUDA(cudaMemcpy(cudaAnchors,anchors.data(),anchorNum*2* sizeof(int),cudaMemcpyHostToDevice));
    _initialized = true;
    return 0;
}
void YOLOPlugin::terminate() {
    if (!_initialized) {
        return;
    }
    cudaFree(cudaAnchors);
    _initialized = false;
}

YOLOPlugin::~YOLOPlugin() {
    terminate();
}
inline bool is_CHW(nvinfer1::Dims const& dims) {
    return (dims.nbDims == 3 &&
            dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
            dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
            dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}
nvinfer1::Dims YOLOPlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) {
    assert(index == 0);
    assert(inputDims);
    assert(nbInputs == 1);
    assert(index == 0);
    nvinfer1::Dims const& input = inputDims[0];
    assert(is_CHW(input));
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    for( int d=0; d<input.nbDims; ++d ) {
        output.type[d] = input.type[d];
        output.d[d] = input.d[d];
    }
    output.d[0] = 7;
    return output;
}
size_t YOLOPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int YOLOPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                        cudaStream_t stream) {
    nvinfer1::Dims input_dims = this->getInputDims(0);
    nvinfer1::DataType type = this->getDataType();
    const int H = input_dims.d[1];
    const int W = input_dims.d[2];
    const int num = batchSize*anchorNum*H*W;
    const float* input_data = static_cast<const float*>(inputs[0]);
    float* out_data= static_cast<float*>(outputs[0]);
    yololayer(stream,num,input_data,out_data,cudaAnchors,anchorNum,classes,H,W,downStride,inferThresh);
    return 0;
}
