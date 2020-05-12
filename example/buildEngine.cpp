//
// Created by yoloCao on 20-5-1.
//

#include <trt.h>
#include <memory>
#include <getopt.h>
#include <string>
int main(int argc, char* argv[])
{

    int opt = 0,option_index = 0;
    static struct option opts[]={{"input-onnx",required_argument, nullptr,'i'},
                          {"output-engine",required_argument,nullptr,'o'},
                          {"batch-size",required_argument,nullptr,'b'},
                          {"model",required_argument,nullptr,'m'},
                          {"calib-file",required_argument,nullptr,'c'},
                          {0,0,0,0}};
    int batchsize = 1;
    std::string onnx = "model/yolov4.onnx";
    std::string engine = "model/yolov4.engine";
    std::string calib = "";
    yolodet::RUN_MODE mode = yolodet::RUN_MODE::FLOAT32;
    while((opt = getopt_long_only(argc,argv,"i:o:b:m:c:",opts,&option_index))!= -1)
    {
        switch (opt){
            case 'i': onnx = std::string(optarg);
                break;
            case 'o': engine = std::string(optarg);
                break;
            case 'b': batchsize = atoi(optarg);
                break;
            case 'c': calib = std::string(optarg);
                break;
            case 'm': {int a=atoi(optarg);
                    switch (a){
                        case 0:mode = yolodet::RUN_MODE::FLOAT32;
                            break;
                        case 1:mode = yolodet::RUN_MODE::FLOAT16;
                            break;
                        case 2:mode = yolodet::RUN_MODE::INT8;
                            break;
                        default:
                            break;
                    };
                    break;}
            default:
                break;
        }
    }
    std::cout<<"input-onnx: "<< onnx << std::endl
            <<"output-engine: "<< engine << std::endl
            <<"max-batchsize: "<< batchsize << std::endl
            <<"calib-file: "<< calib << std::endl;
    yolodet::yoloNet net(onnx,calib,batchsize,mode);
    net.saveEngine(engine);
    std::cout << "save " <<engine << std::endl;
}
