//
// Created by yoloCao on 20-5-1.
//

#ifndef TENSORRT_YOLOV4_TRT_H
#define TENSORRT_YOLOV4_TRT_H


#include <string>
#include <memory>
#include "NvInfer.h"
#include <vector>
#include <opencv2/opencv.hpp>
namespace yolodet{
    using namespace std;
    enum class RUN_MODE
    {
        FLOAT32 = 0 ,
        FLOAT16 = 1 ,
        INT8    = 2
    };
    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };

    class yoloNet{
    template <typename T>
    using nvUniquePtr = unique_ptr<T, InferDeleter>;
    public:
        yoloNet(const string &onnxFile, const string &calibFile,int maxBatchSzie,RUN_MODE mode = RUN_MODE::FLOAT32);
        yoloNet(const string &engineFile);
        yoloNet()= delete;
        ~yoloNet();
        bool saveEngine(const std::string& fileName);
        bool initEngine();
        bool infer(const cv::Mat &img, void* outputData);
        shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
        shared_ptr<nvinfer1::IExecutionContext> mContext;
        nvinfer1::IPluginFactory *mPlugin;
        cudaStream_t mCudaStream;
        vector<void *> mCudaBuffers;
        void *mCudaImg;
        vector<size_t > mBindBufferSizes;
        nvinfer1::Dims inputDim;
    };


}

#endif //TENSORRT_YOLOV4_TRT_H
