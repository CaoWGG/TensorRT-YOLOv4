//
// Created by yoloCao on 20-5-11.
//

#ifndef TENSORRT_YOLOV4_UTILS_H
#define TENSORRT_YOLOV4_UTILS_H
#include "NvInfer.h"


class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportableSeverity;
};

bool doNms(void* bboxs,int w,int h,int in_w,int in_h,float nmsThresh,bool keepration ,bool keepcenter){
    typedef struct  {
        float x1;
        float y1;
        float x2;
        float y2;
        float conf;
        float cls;
        float batchId;
    } detection ;
    auto computeIOU = [](detection &bb1,detection &bb2){

        float inter_x1 = fmax(bb1.x1,bb2.x1);
        float inter_y1 = fmax(bb1.y1,bb2.y1);
        float inter_x2 = fmin(bb1.x2,bb2.x2);
        float inter_y2 = fmin(bb1.y2,bb2.y2);
        if (inter_x1 > inter_x2 || inter_y1 > inter_y2) return 0.f;
        float interArea = (inter_y2 - inter_y1) * (inter_x2 - inter_x1);
        float unioArea = (bb1.x2 - bb1.x1) * (bb1.y2 - bb1.y1) + (bb2.x2 - bb2.x1) * (bb2.y2 - bb2.y1) ;
        return interArea/(unioArea -interArea);
    };
    float scaleX = (w*1.0f / in_w);
    float scaleY = (h*1.0f / in_h);
    float shiftX = 0.f ,shiftY = 0.f;
    if(keepration)scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;
    if(keepration && keepcenter){shiftX = (in_w - w/scaleX)/2.f;shiftY = (in_h - h/scaleY)/2.f;}
    std::vector<int> clsId(80,-1);
    std::vector<std::vector<detection >> clsBboxs;
    int numdDet = ((float*)bboxs)[0];
    float*  temp= (float*)bboxs + 1;
    int numCls = 0;
    for(int i = 0; i< numdDet; ++i){
        float x = temp[0],y = temp[1],w = temp[2],h = temp[3];
        int cls = temp[4];
        float conf = temp[5];
        int batchId = temp[6];
        float x1 = (x - w/2.f - shiftX + 0.5)*scaleX - 0.5;
        float y1 = (y - h/2.f - shiftY + 0.5)*scaleY - 0.5;
        float x2 = (x + w/2.f - shiftX + 0.5)*scaleX - 0.5;
        float y2 = (y + h/2.f - shiftY + 0.5)*scaleY - 0.5;
        if(clsId[cls] == -1){clsId[cls]=numCls;numCls++;clsBboxs.resize(numCls);}
        clsBboxs[clsId[cls]].push_back({x1,y1,x2,y2,conf,float(cls),float(batchId)});
        temp+=7;
    }
    numdDet = 0;
    temp= (float*)bboxs + 1;

    for(int i = 0; i < numCls;++i){
        auto& clsbb = clsBboxs[i];
        sort(clsbb.begin(),clsbb.end(),[=](const detection& bb1,const detection& bb2){
            return bb1.conf > bb2.conf;
        });
        for(size_t j = 0; j < clsbb.size(); ++j) {
            auto &bestBB = clsbb[j];
            for(size_t n = j + 1;n < clsbb.size() ; ++n)
            {
                if (computeIOU(bestBB,clsbb[n]) > nmsThresh)
                {
                    clsbb.erase(clsbb.begin()+n);
                    --n;
                }
            }
        }
        mempcpy(temp + numdDet*7,clsbb.data(),clsbb.size()*sizeof(float)*7);
        numdDet+=clsbb.size();
    }
    ((float*)bboxs)[0] = numdDet;
    return 1;
}
#endif //TENSORRT_YOLOV4_UTILS_H
