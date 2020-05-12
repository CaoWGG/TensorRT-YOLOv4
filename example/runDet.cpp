//
// Created by yoloCao on 20-5-5.
//

#include <trt.h>
#include <memory>
#include <getopt.h>
#include <string>
#include <chrono>

int main(int argc,  char* argv[]){
    int opt = 0,option_index = 0;
    static struct option opts[]={{"input-engine",required_argument, nullptr,'i'},
                                 {"image",required_argument,nullptr,'p'},
                                 {"video",required_argument,nullptr,'v'},
                                 {0,0,0,0}};
    std::string engine = "model/yolov4.engine";
    std::string image = "dog.jpg";
    std::string video = "nuscenes_mini.mp4";
    while((opt = getopt_long_only(argc,argv,"i:p:v:",opts,&option_index))!= -1)
    {
        switch (opt){
            case 'i': engine = std::string(optarg);
                break;
            case 'p': image = std::string(optarg);
                break;
            case 'v': video = std::string(optarg);
                break;
            default:
                break;
        }
    }
    std::cout<<"input-engine: "<< engine << std::endl
             <<"image: "<< image << std::endl
             <<"video: "<< video << std::endl;

    cv::namedWindow("result",cv::WINDOW_NORMAL);
    cv::resizeWindow("result",1024,768);

    yolodet::yoloNet net(engine);
    std::unique_ptr<float[]> outputData(new float[net.mBindBufferSizes[1]]);

    cv::Mat img ;
    int i=0;
    double runtime=0;
    char showText[128] = {0};
    int base_line ;
    if(image.size()>0)
    {

        img = cv::imread(image);
        auto start = std::chrono::system_clock::now();
        net.infer(img, outputData.get());
        auto end = std::chrono::system_clock::now();
        i++;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        runtime+=double(duration.count());
        std::cout <<  "cost: "
                  << runtime/i << "  ms"<< std::endl;

        //show img
        float label_scale = img.rows * 0.0009;
        int box_think = (img.rows+img.cols) * .001 ;
        int num_det = static_cast<int>(outputData[0]);
        float *temp = outputData.get() + 1;
        for (int  j=0; j<num_det ; ++j){
            cv::rectangle(img,cv::Point(temp[0],temp[1]),cv::Point(temp[2],temp[3]),cv::Scalar(255,0,0),3);
            sprintf(showText,"cls_%d : %0.2f",int(temp[5]),temp[4]);
            auto size = cv::getTextSize(showText,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);
            cv::putText(img,showText,
                        cv::Point(temp[0],temp[1] - size.height),
                        cv::FONT_HERSHEY_COMPLEX, label_scale , cv::Scalar(0,0,255), box_think*2/3, 8, 0);
            temp+=7;
        }
        cv::imshow("result",img);
        if((cv::waitKey(0)& 0xff) == 27){
            cv::destroyAllWindows();
            return 0;
        };

    }

    if(video.size()>0){
        cv::VideoCapture cap(video);
        while (cap.read(img))
        {

            auto start = std::chrono::system_clock::now();
            net.infer(img, outputData.get());
            auto end = std::chrono::system_clock::now();
            i++;
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            runtime+=double(duration.count());
            std::cout <<  "cost: "
                      << runtime/i << "  ms" <<std::endl;


            /// show img
            float label_scale = img.rows * 0.0009;
            int box_think = (img.rows+img.cols) * .001 ;
            int num_det = static_cast<int>(outputData[0]);
            float *temp = outputData.get() + 1;
            for (int  j=0; j<num_det ; ++j){
                cv::rectangle(img,cv::Point(temp[0],temp[1]),cv::Point(temp[2],temp[3]),cv::Scalar(255,0,0),3);
                sprintf(showText,"cls_%d : %0.2f",int(temp[5]),temp[4]);
                auto size = cv::getTextSize(showText,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);
                cv::putText(img,showText,
                            cv::Point(temp[0],temp[1] - size.height),
                            cv::FONT_HERSHEY_COMPLEX, label_scale , cv::Scalar(0,0,255), box_think*2/3, 8, 0);
                temp+=7;
            }
            cv::imshow("result",img);
            if((cv::waitKey(0)& 0xff) == 27){
                cv::destroyAllWindows();
                return 0;
            };
        }

    }

    return 0;
}
