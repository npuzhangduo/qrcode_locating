#include<iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ostream>
#include <fstream>
#include <sstream>
using namespace std;
enum { THRESH_BINARY=0, THRESH_BINARY_INV=1 };



class MyQRCodeLocating {
public:
    MyQRCodeLocating();
    ~MyQRCodeLocating();

    bool imageProcess(cv::Mat &src,cv::Mat &res);

    void cornerDetect();
private:

    void myAdaptiveThreshold( cv::Mat &src, cv::Mat &dst, double maxValue,int type,int blockSize, double delta );
    bool paintingBox(vector<vector<cv::Point> > &contours,vector<cv::Vec4i> &hierarchy);
    bool mainProcess(bool big);

    cv::Mat srcImage;
    cv::Mat imageAfterResize;
    cv::Mat greyImage;
    cv::Mat threImage;
    cv::Mat resImage;
    
    cv::Mat threeContoursImage;

    vector<vector<cv::Point> >contours;
    vector<cv::Vec4i> hierarchy;

    vector<vector<cv::Point> >threeContours;
};