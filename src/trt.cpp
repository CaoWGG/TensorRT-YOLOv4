//
// Created by yoloCao on 20-5-1.
//

#include <trt.h>
#include <NvOnnxParser.h>
#include <NvOnnxParserRuntime.h>
#include <fstream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <NvOnnxConfig.h>
#include "resize.h"
#include <chrono>
#include "utils.h"
#include <numeric>


static Logger gLogger;
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << cudaGetErrorString(error_code) << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

namespace yolodet{
    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            case nvinfer1::DataType::kINT8: return 1;

        }
        throw std::runtime_error("Invalid DataType.");
        return 0;
    }
    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }
    inline void* safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
            std::cerr << "Out of memory" << std::endl;
            exit(1);
        }
        return deviceMem;
    }
    yoloNet::yoloNet(const std::string &onnxFile, const std::string &calibFile, int maxBatchSzie,yolodet::RUN_MODE mode) {
        cudaSetDevice(0);
        auto builder = nvUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        assert(builder!= nullptr);

        auto network =  nvUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());

        assert(network!= nullptr);

        mPlugin =nvonnxparser::createPluginFactory(gLogger);

        auto parser = nvUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        assert(parser!= nullptr);

        if (!parser->parseFromFile(onnxFile.c_str(), 2))
        {
            std::string msg("failed to parse onnx file");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }


        builder->setMaxBatchSize(maxBatchSzie);
        builder->setMaxWorkspaceSize(1 << 30);// 1G

        if (mode==RUN_MODE::FLOAT16)
        {
            std::cout <<"setFp16Mode"<<std::endl;
            if (!builder->platformHasFastFp16())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
            builder->setFp16Mode(true);
        }
        if (mode==RUN_MODE::INT8)
        {
            std::cout <<"setInt8Mode"<<std::endl;
            if (!builder->platformHasFastInt8())
                std::cout << "Notice: the platform do not has fast for int8" << std::endl;
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(nullptr);
        }
        std::cout << "Begin building engine..." << std::endl;
        mEngine = shared_ptr<nvinfer1::ICudaEngine>(
                builder->buildCudaEngine(*network), InferDeleter());
        if (!mEngine){
            std::string error_message ="Unable to create engine";
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
            exit(-1);
        }
        std::cout << "End building engine..." << std::endl;
        shared_ptr<nvinfer1::IHostMemory> modelStream(mEngine->serialize(),InferDeleter());
        assert(modelStream != nullptr);
        shared_ptr<nvinfer1::IRuntime> mRunTime(nvinfer1::createInferRuntime(gLogger),InferDeleter());
        assert(mRunTime != nullptr);
        mEngine= shared_ptr<nvinfer1::ICudaEngine>(mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(),mPlugin),InferDeleter());
        assert(mEngine != nullptr);
        assert(initEngine());

    }
    yoloNet::yoloNet(const std::string &engineFile)
    {
        cudaSetDevice(0);
        using namespace std;
        fstream file;

        file.open(engineFile,ios::binary | ios::in);
        if(!file.is_open())
        {
            cout << "read engine file" << engineFile <<" failed" << endl;
            return;
        }
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();


        shared_ptr<nvinfer1::IRuntime> mRunTime(nvinfer1::createInferRuntime(gLogger),InferDeleter());
        assert(mRunTime != nullptr);
        mPlugin = nvonnxparser::createPluginFactory(gLogger);

        mEngine = shared_ptr<nvinfer1::ICudaEngine>(mRunTime->deserializeCudaEngine(data.get(), length, mPlugin),InferDeleter());
        assert(mEngine != nullptr);
        assert(initEngine());
    }

    bool yoloNet::initEngine(){
        const int maxBatchSize= mEngine->getMaxBatchSize();
        mContext = shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(),InferDeleter());
        assert(mContext != nullptr);
        int nbBindings = mEngine->getNbBindings();
        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(2);
        for(int i = 0; i < nbBindings; ++i){
            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            int totalSize = maxBatchSize*volume(mEngine->getBindingDimensions(i)) * getElementSize(dtype);
            if(i == 0){
                mBindBufferSizes[0] = totalSize;
                mCudaBuffers[0] = safeCudaMalloc(totalSize);
                mBindBufferSizes[1] = 0;
            } else mBindBufferSizes[1] += totalSize;
        }
        mCudaBuffers[1] = safeCudaMalloc(mBindBufferSizes[1]);
        for (int i = 2; i < nbBindings; ++i) mCudaBuffers[i]=mCudaBuffers[1];
        mCudaImg =  safeCudaMalloc(4096*4096*3* sizeof(uchar)); // max input image shape
        CUDA_CHECK(cudaStreamCreate(&mCudaStream));
        inputDim = mEngine->getBindingDimensions(0);
        return 1;
    }
    yoloNet::~yoloNet() {
        cudaStreamSynchronize(mCudaStream);
        cudaStreamDestroy(mCudaStream);
        if(mCudaBuffers[0])CUDA_CHECK(cudaFree(mCudaBuffers[0]));
        if(mCudaBuffers[1])CUDA_CHECK(cudaFree(mCudaBuffers[1]));
        if(mCudaImg)CUDA_CHECK(cudaFree(mCudaImg));
    }

    bool yoloNet::saveEngine(const std::string &fileName){
        if(mEngine)
        {
            shared_ptr<nvinfer1::IHostMemory> data(mEngine->serialize(),InferDeleter());
            std::ofstream file;
            file.open(fileName,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                return 0;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
        }
        return 1;
    }
    bool yoloNet::infer(const cv::Mat &img, void *outputData) {
        bool keepRation = 1 ,keepCenter= 1;
        CUDA_CHECK(cudaMemcpy(mCudaImg,img.data,img.step[0]*img.rows,cudaMemcpyHostToDevice));
        resizeAndNorm(mCudaImg,(float*)mCudaBuffers[0],img.cols,img.rows,inputDim.d[2],inputDim.d[1],keepRation,keepCenter,0);
        CUDA_CHECK(cudaMemset(mCudaBuffers[1],0, sizeof(int)));
        mContext->execute(1,&mCudaBuffers[0]);
        float det=0;
        CUDA_CHECK(cudaMemcpy(&det,mCudaBuffers[1], sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outputData,mCudaBuffers[1], sizeof(float) + int(det)*7* sizeof(float),cudaMemcpyDeviceToHost));
        doNms(outputData,img.cols,img.rows,inputDim.d[2],inputDim.d[1],0.45,keepRation,keepCenter);
        return 1;
    }
}