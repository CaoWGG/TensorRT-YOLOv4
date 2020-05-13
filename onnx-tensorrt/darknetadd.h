//
// Created by yoloCao on 20-5-12.
//

#ifndef TENSORRT_YOLOV4_DARKNETADD_H
#define TENSORRT_YOLOV4_DARKNETADD_H
#pragma once
#include "plugin.hpp"
#include "serialize.hpp"
#include <vector>
#include <cuda_runtime.h>

class ADDPlugin final : public onnx2trt::Plugin {

    bool _initialized;
protected:
    void deserialize(void const* serialData, size_t serialLength) {
        deserializeBase(serialData, serialLength);
    }
    size_t getSerializationSize() override {
        return  getBaseSerializationSize();
    }
    void serialize(void *buffer) override {
        serializeBase(buffer);
    }
public:
    ADDPlugin();

    ADDPlugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
    }
    const char* getPluginType() const override { return "DarkNetAdd"; }

    int getNbOutputs() const override { return 1; }
    nvinfer1::Dims getOutputDimensions(int index,
                                       const nvinfer1::Dims *inputDims,
                                       int nbInputs) override;
    int initialize() override;
    void terminate() override;
    int enqueue(int batchSize,
                const void *const *inputs, void **outputs,
                void *workspace, cudaStream_t stream) override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    ~ADDPlugin();
};
#endif //TENSORRT_YOLOV4_DARKNETADD_H
