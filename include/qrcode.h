#include<iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ostream>
#include <fstream>
#include <sstream>
using namespace std;
enum { THRESH_BINARY=0, THRESH_BINARY_INV=1 };



class MyQRCode {
public:
    MyQRCode();
    ~MyQRCode();

    bool imageProcess(cv::Mat &src,cv::Mat &res,cv::Mat &PersRes,vector<vector<int> > &Code);


private:

    void myAdaptiveThreshold( cv::Mat &src, cv::Mat &dst, double maxValue,int type,int blockSize, double delta );
    bool paintingBox(vector<vector<cv::Point> > &contours,vector<cv::Vec4i> &hierarchy);
    bool mainProcess(bool big);
    void Decode();

    cv::Mat srcImage;
    cv::Mat imageAfterResize;
    cv::Mat greyImage;
    cv::Mat threImage;
    cv::Mat resImage;
    
    cv::Mat threeContoursImage;

    cv::Mat PerspectiveImage;

    vector<vector<cv::Point> >contours;
    vector<cv::Vec4i> hierarchy;

    vector<vector<int> >resCode;

    vector<vector<cv::Point> >threeContours;
};