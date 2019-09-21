
#include "qrcode.h"
using namespace std;



MyQRCodeLocating::MyQRCodeLocating() {

}

MyQRCodeLocating::~MyQRCodeLocating() {

}

bool MyQRCodeLocating::imageProcess(cv::Mat &src,cv::Mat &res) {
    srcImage = src.clone();
    bool find = mainProcess(true);
    if (!find)
        mainProcess(false);
    res = resImage.clone();
    return find;
}

void MyQRCodeLocating::myAdaptiveThreshold( cv::Mat &src, cv::Mat &dst, double maxValue, int type,int blockSize, double delta ) {
    cv::Size size = src.size();
    cv::Mat mean;

    if( src.data != dst.data )
        mean = dst;

    cv::GaussianBlur( src, mean, cv::Size(blockSize, blockSize), 0, 0, cv::BORDER_REPLICATE );
    int i, j;
    uchar imaxval = cv::saturate_cast<uchar>(maxValue);
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);

    if( src.isContinuous() && mean.isContinuous() && dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }
    
    for( i = 0; i < size.height; i++ )
    {
        const uchar* sdata = src.data + src.step*i;
        const uchar* mdata = mean.data + mean.step*i;
        uchar* ddata = dst.data + dst.step*i;

        for( j = 0; j < size.width; j++ ) {
            if (type == CV_THRESH_BINARY) {
                if (sdata[j] - mdata[j] < -idelta) 
                    ddata[j] = 0;
                else
                    ddata[j] = 255;
            }
            else {
                if (sdata[j] - mdata[j] < -idelta) 
                    ddata[j] = 255;
                else
                    ddata[j] = 0;
            }
            
        }
    }
}


bool MyQRCodeLocating::paintingBox(vector<vector<cv::Point> > &contours,vector<cv::Vec4i> &hierarchy) {
    vector<cv::Moments>mu(contours.size());
    for (int i = 0; i < contours.size(); i++){
        mu[i] = moments(contours[i], false);
    }
    
    vector<cv::Point2f>mc(contours.size());

    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
    vector<cv::Point2f> contoursCenter;
    for (int i = 0; i < contours.size(); i++)
    {
        int x = 0;
        int y = 0;
        for (int j = 0;j < contours[i].size();j++) {
            x += contours[i][j].x;
            y += contours[i][j].y;
        }
        int xx = x / contours[i].size();
        int yy = y / contours[i].size();
        cv::Point tmp(xx,yy);
        contoursCenter.push_back(tmp);
    }

    vector<vector<cv::Point> > finalContours;
    vector<cv::Point2f> center;


    cv::Mat debug1 = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
    for (int i = 0;i < contours.size();i++) {
            cv::drawContours( debug1, contours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    //cv::imshow("debug1", debug1);
    // 过滤同心的轮廓
    if (contours.size() != 3) {

        vector<int> delete_list;
        for (int i = 0;i < contours.size();i++) {
            for (int j = i+1;j < contours.size();j++) {
                double dis = sqrt((contoursCenter[i].x - contoursCenter[j].x) * (contoursCenter[i].x - contoursCenter[j].x)
                                    +(contoursCenter[i].y - contoursCenter[j].y) * (contoursCenter[i].y - contoursCenter[j].y));
                //cout <<dis<<endl;
                if (dis < 20) {
                    delete_list.push_back(j);
                }
            }
        }

        for (int i = 0;i < contours.size();i++) {
            int flag = 0;
            for (int j = 0;j < delete_list.size();j++) {
                if (i == delete_list[j])
                    flag = 1;
            }
            if (flag == 0) {
                finalContours.push_back(contours[i]);
                center.push_back(mc[i]);
            }
        }
    }
    else
    {
        finalContours = contours;
        center = mc;
    }
    cv::Mat debug = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
    for (int i = 0;i < finalContours.size();i++) {
            cv::drawContours( debug, finalContours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    //cv::imshow("debug", debug);

    vector<double> contoursArea;
    for (int i = 0;i < finalContours.size();i++) {
        double tmp = cv::contourArea(finalContours[i]);
        contoursArea.push_back(tmp);
    }

    double sum = accumulate(contoursArea.begin(),contoursArea.end(), 0.0);  
    double mean =  sum / contoursArea.size(); //均值  
  
    double accum  = 0.0;  
    for_each (contoursArea.begin(),contoursArea.end(), [&](const double d) {  
        accum  += (d-mean)*(d-mean); });  
  
    double stdev = sqrt(accum/(contoursArea.size()-1)); //方差
    //cout <<stdev<<endl;
    if (finalContours.size() > 3 && stdev > 1500) {
        int index1 = 0;
        double min1 = 100000;
        int index2 = 1;
        double min2 = 100000;
        int index3 = 2;
        double min3 = 100000;
        for (int i = 0;i < finalContours.size();i++) {
            double diff = fabs(contoursArea[i] - mean);
            
            if (diff <= min1) {
                min3 = min2;
                min2 = min1;
                min1 = diff;
                index3 = index2;
                index2 = index1;
                index1 = i;
            }
            else if(diff <= min2) {
                min3 = min2;
                min2 = diff;
                index3 = index2;
                index2 = i;
            }
            else if (diff <= min3) {
                min3 = diff;
                index3 = i;
            }
        }
        vector<vector<cv::Point> > tmpContours;
        vector<cv::Point2f> tmpcenter;
        tmpContours.push_back(finalContours[index1]);
        tmpContours.push_back(finalContours[index2]);
        tmpContours.push_back(finalContours[index3]);
        tmpcenter.push_back(center[index1]);
        tmpcenter.push_back(center[index2]);
        tmpcenter.push_back(center[index3]);

        finalContours = tmpContours;
        center = tmpcenter;
    }
    
    double dis[10][10];
    for (int i = 0;i < 10;i++)
        for (int j = 0;j < 10;j++)
            dis[i][j] = 0;
    for (int i = 0;i < finalContours.size();i++) {
        for (int j = 0;j < finalContours.size();j++) {
            if (i == j)
                continue;
            if (dis[j][i] != 0) {
                dis[i][j] = dis[j][i];
            }
            else {
                dis[i][j] = sqrt((center[i].x - center[j].x) * (center[i].x - center[j].x)
                            + (center[i].y - center[j].y) * (center[i].y - center[j].y));   
            }
        }
    }
    contours.clear();
    mc.clear();
     
    vector<cv::Point2f> threeCenter;
    if (finalContours.size() > 3)
    {
        int index1 = 0;
        double min1 = 100000;
        int index2 = 1;
        double min2 = 100000;
        int index3 = 2;
        double min3 = 100000;
        for (int i = 0;i < finalContours.size();i++) {
            double sum = 0;
            for (int j = 0;j < finalContours.size();j++) {
                sum += dis[i][j];
            }
            
            if (sum <= min1) {
                min3 = min2;
                min2 = min1;
                min1 = sum;
                index3 = index2;
                index2 = index1;
                index1 = i;
            }
            else if(sum <= min2) {
                min3 = min2;
                min2 = sum;
                index3 = index2;
                index2 = i;
            }
            else if (sum <= min3) {
                min3 = sum;
                index3 = i;
            }
        }
        
        for (int j = 0;j < finalContours.size();j++) {
            if (j == index1 || j == index2 || j == index3) {
                threeContours.push_back(finalContours[j]);
            }
        }
        
        threeCenter.push_back(center[index1]);
        threeCenter.push_back(center[index2]);
        threeCenter.push_back(center[index3]);
        //cout <<"debug ll"<<endl;
    }
    else
    {
        threeContours = finalContours;
        threeCenter = center;
    }
    finalContours.clear();
    center.clear();
   

    cv::Mat drawing_three = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
    for (int i = 0;i < threeContours.size();i++) {
            cv::drawContours( drawing_three, threeContours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    // cv::imshow("three contours",drawing_three);
    cv::namedWindow("result",1);
    
    threeContoursImage = drawing_three.clone();
    if (threeContours.size() == 3) {


        double dis[3][3];
        for (int i = 0;i < 3;i++)
            for (int j = 0;j < 3;j++)
                dis[i][j] = 0;
        for (int i = 0;i < threeContours.size();i++) {
            for (int j = 0;j < threeContours.size();j++) {
                if (i == j)
                    continue;
                if (dis[j][i] != 0) {
                    dis[i][j] = dis[j][i];
                }
                else {
                    dis[i][j] = sqrt((threeCenter[i].x - threeCenter[j].x) * (threeCenter[i].x - threeCenter[j].x)
                                + (threeCenter[i].y - threeCenter[j].y) * (threeCenter[i].y - threeCenter[j].y));   
                }
            }
        }
        int index_con = 0;
        double min = 10000;
        for (int i = 0;i < threeContours.size();i++) {
            double sum = 0;
            for (int j = 0;j < threeContours.size();j++) {
                sum += dis[i][j];
            }
            if (sum < min) {
                min = sum;
                index_con = i;
            }
        }
        cv::Mat drawing_t = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
        

        cv::Point fourth_point(0,0);
        for (int i = 0;i < threeCenter.size();i++) {
            if (i == index_con) {
                fourth_point.x -= threeCenter[i].x;
                fourth_point.y -= threeCenter[i].y;
            }
            else {
                fourth_point.x += threeCenter[i].x;
                fourth_point.y += threeCenter[i].y;
            }
        }
        for (int i = 0;i < threeContours.size();i++) {
            cv::drawContours( drawing_t, threeContours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
        }
        
        for (int i = 0;i < threeCenter.size();i++) {
            drawing_t.at<uchar>(threeCenter[i].y,threeCenter[i].x) = 255;
        }

        drawing_t.at<uchar>(fourth_point.y,fourth_point.x) = 255;
        cv::Point final_corner[4];
        final_corner[3] = fourth_point;
        final_corner[1] = threeCenter[index_con];
        int two = 0;
        for (int i = 0;i < threeCenter.size();i++) {
            if (i != index_con) {
                if (two == 0) {
                    final_corner[0] = threeCenter[i];
                    two ++;
                }
                else {
                    final_corner[2] = threeCenter[i];
                    break;
                }
            }
        }

        vector<cv::Point> rec_4;
        cv::Point diff;
        diff.x = fourth_point.x - center[index_con].x;
        diff.y = fourth_point.y - center[index_con].y;

        for (int i = 0;i < threeContours[index_con].size();i++) {
            double x = threeContours[index_con][i].x + diff.x;
            double y = threeContours[index_con][i].y + diff.y;
            cv::Point tmp(x,y);
            rec_4.push_back(tmp);
        }
        vector<vector<cv::Point> > fouth_contour;
        fouth_contour.push_back(rec_4);
        cv::drawContours( drawing_t, fouth_contour, 0 ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );

        
        for(int i = 0;i < 4;i++) {
            cv::line(drawing_t, final_corner[i], final_corner[(i+1)%4], cv::Scalar(255), 2);
        }
        //cv::imshow("drawing_t",drawing_t);
        vector<vector<cv::Point> > singleContours;

        cv::findContours(drawing_t,singleContours,hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        

        resImage = imageAfterResize.clone();
        for (int i = 0;i < singleContours.size();i++) {
            cv::RotatedRect rect = cv::minAreaRect(singleContours[i]);
		    cv::Point2f boxpoint[4];
		    rect.points(boxpoint);
		

            for (int j = 0; j < 4; j++)
			    cv::line(resImage, boxpoint[j], boxpoint[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
            
            
            
        }    
        //cv::imshow("result",resImage);
        return true;
        
    }
    else
    {
        //cv::imshow("result",imageAfterResize);
        resImage = imageAfterResize.clone();
        return false;
    }
}


bool MyQRCodeLocating::mainProcess(bool big) {
    // 太大图片，缩小图像的分辨率
    if (srcImage.cols > 1500) {
            
        int nRows = 640;
        int nCols = srcImage.cols*640/ srcImage.rows;
        cv::Mat dst(nRows, nCols, srcImage.type());
        resize(srcImage,dst,dst.size(),0,0, cv::INTER_LINEAR);
        imageAfterResize = dst.clone();
    }
    else
    {
        imageAfterResize = srcImage.clone();    
    }
    cv::imshow("image after resize",imageAfterResize);
    cv::cvtColor(imageAfterResize,greyImage,CV_RGB2GRAY);

    // 高斯模糊，但如果目标太小会造成不良的影响
    if (big)
        cv::GaussianBlur(greyImage,greyImage,cv::Size(3,3),1.5);
    cv::Mat ad_thre(greyImage.rows,greyImage.cols,greyImage.type());

    myAdaptiveThreshold(greyImage,ad_thre, 255,cv::THRESH_BINARY_INV, 93, 8);
    threImage = ad_thre.clone();

    //cv::imshow("threhold image",threImage);


    cv::findContours(threImage,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
    
    cv::Mat drawing = cv::Mat::zeros(greyImage.size(),CV_8U);
    vector<vector<cv::Point> >contoursAfterFilter;

    for (int i = 0;i < contours.size();i++) {
        cv::drawContours( drawing, contours, i,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    //cv::imshow("original contours",drawing);


    cv::Mat drawingAfterFilter = cv::Mat::zeros(greyImage.size(),CV_8U);
    // 过滤无用轮廓
    for( int i = 0; i< contours.size(); i++ )
    {
        if (contours[i].size() < 5)
            continue;
        if (contours[i].size() > 150)
            continue;

        
        if (hierarchy[i][2] == -1)
            continue;
        double kid = hierarchy[i][2];

        if (hierarchy[kid][2] == -1)
            continue;

        cv::RotatedRect rRect = cv::minAreaRect(contours[i]);
        double contoursArea = cv::contourArea(contours[i]);
        double rectArea = rRect.size.height * rRect.size.width;
        
        if (contoursArea > 15000)
            continue;
        if (contoursArea / rectArea > 1.3 || contoursArea / rectArea < 0.7)
            continue;
       
        double kid_kid = hierarchy[kid][2];
        contoursAfterFilter.push_back(contours[i]);
        cv::drawContours( drawingAfterFilter, contours, i,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    //cv::imshow("contoursAfterFilter",drawingAfterFilter);
    bool find;
    find = paintingBox(contoursAfterFilter,hierarchy);
    return find;
}


void MyQRCodeLocating::cornerDetect() {
    cv::Mat drawing_three = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
    for (int i = 0;i < threeContours.size();i++) {
            cv::drawContours( drawing_three, threeContours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    cv::imshow("three contours",drawing_three);

    cv::Mat dst = cv::Mat::zeros(threeContoursImage.size(),CV_8U);
 
}

