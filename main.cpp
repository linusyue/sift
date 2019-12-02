#include <iostream>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <dirent.h>
#include "sift.h"

using namespace cv;
using namespace std;

void usage(char* argv[])
{
    //Folder struct
    //-directory/
    //     --image2/
    //     --image3/
    std::cout << "usage: " << argv[0] << "sift [directory]" << std::endl;

    cout<<"Foler structure"<<endl;
    cout<<"-[directory]/"<<endl;
    cout<<"      --image2/"<<endl;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        usage(argv);
        return -1;
    }

    string directory = argv[1];

    DIR *dp;
    struct dirent *ep;

    string image2_dir;
    image2_dir = directory + "/" + "image_2";
    dp = opendir(image2_dir.c_str());

    if (dp == NULL) {
        std::cerr << "Invalid folder structure under: " << directory << std::endl;
        usage(argv);
        exit(EXIT_FAILURE);
    }

    string I0_path;
    string I1_path;

    while ((ep = readdir(dp)) != NULL) 
    {
        // Skip directories
        if (!strcmp (ep->d_name, "."))
            continue;
        if (!strcmp (ep->d_name, ".."))
            continue;

        string postfix = "_10.png";
        string::size_type idx;

        string image0_name = ep->d_name;

        //find _10 
        idx = image0_name.find(postfix);

        if(idx == string::npos )
            continue;  

        string image1_name = image0_name;

        image1_name.replace(8, 1, "1");

        I0_path = directory + "/" + "image_2" + "/" + image0_name;
        I1_path = directory + "/" + "image_2" + "/" + image1_name;

        cout<<"I0: "<<I0_path<<endl;
        cout<<"I1: "<<I1_path<<endl;

        Mat I0 = imread(I0_path);
        Mat I1 = imread(I1_path);

        if (I0.empty() || I1.empty())
        {
            std::cerr << "failed to read any image." << std::endl;
            break;
        }

        CV_Assert(I0.size() == I1.size() && I0.type() == I1.type());

        //convert to gray
        Mat I0_Gray, I1_Gray;
        cvtColor(I0, I0_Gray, cv::COLOR_BGR2GRAY);
        cvtColor(I1, I1_Gray, cv::COLOR_BGR2GRAY);

        imshow("I0", I0);
        imshow("I1", I1);


        //sift with opencv
        cv::Ptr<cv::xfeatures2d::SIFT> cvsift = cv::xfeatures2d::SIFT::create();
        vector<cv::KeyPoint> keyPoint0, keyPoint1;
        Mat desp0, desp1;

        //        cvsift->detect(I0, keyPoint0);
        //        cvsift->detect(I1, keyPoint1);
        cvsift->detectAndCompute(I0, cv::Mat(), keyPoint0, desp0);
        cvsift->detectAndCompute(I1, cv::Mat(), keyPoint1, desp1);

        Mat showPoint0, showPoint1;

        drawKeypoints(I0, keyPoint0, showPoint0, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(I1, keyPoint1, showPoint1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        imshow("Opencv keypoint0", showPoint0);

        Sift sift;
        const auto t1 = std::chrono::system_clock::now();

        //Under development
//         sift.compute(I0_Gray);

        const auto t2 = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;

        waitKey();
    }

    return 0; 
}
