
#include "qrcode.h"
using namespace std;



MyQRCode::MyQRCode() {

}

MyQRCode::~MyQRCode() {

}

bool MyQRCode::imageProcess(cv::Mat &src,cv::Mat &res,cv::Mat &PersRes,vector<vector<int> > &Code) {
    srcImage = src.clone();
    bool find = mainProcess(true);
    if (!find) {
        find = mainProcess(false);
    }
    if (find) {
        
        Decode();
        PersRes = PerspectiveImage.clone();
        Code = resCode;
    }
    res = resImage.clone();
    
    return find;
}

void MyQRCode::myAdaptiveThreshold( cv::Mat &src, cv::Mat &dst, double maxValue, int type,int blockSize, double delta ) {
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


bool MyQRCode::paintingBox(vector<vector<cv::Point> > &contours,vector<cv::Vec4i> &hierarchy) {
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


bool MyQRCode::mainProcess(bool big) {
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

double caculateDis(cv::Point2f point1,cv::Point2f point2) {
    double x_diff = point1.x - point2.x;
    double y_diff = point1.y - point2.y;
    return sqrt(x_diff * x_diff + y_diff * y_diff);
}

void MyQRCode::Decode() {


    vector<cv::Point2f> center;
    for (int i = 0; i < threeContours.size(); i++)
    {
        double x = 0;
        double y = 0;
        for (int j = 0;j < threeContours[i].size();j++) {
            x += threeContours[i][j].x;
            y += threeContours[i][j].y;
        }
        double xx = x / threeContours[i].size();
        double yy = y / threeContours[i].size();
        cv::Point tmp(xx,yy);
        center.push_back(tmp);
    }
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
                dis[i][j] = sqrt((center[i].x - center[j].x) * (center[i].x - center[j].x)
                            + (center[i].y - center[j].y) * (center[i].y - center[j].y));   
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
    vector<cv::Point2f> newCenter;
    vector<vector<cv::Point> >newContours; 
    newCenter.push_back(center[index_con]);
    newContours.push_back(threeContours[index_con]);

    for (int i = 0;i < center.size();i++) {
        if (i != index_con) {
            newCenter.push_back(center[i]);
            newContours.push_back(threeContours[i]);
        }
    }
    index_con = 0;
    int index_right = 3;
    int index_down = 3;

    if (newCenter[1].y >= newCenter[0].y && newCenter[2].y >= newCenter[0].y) {
        if (newCenter[1].x > newCenter[2].x) {
            index_right = 1;
            index_down = 2;
        }
        else {
            index_right = 2;
            index_down = 1;
        }
    }
    else if (newCenter[1].y <= newCenter[0].y && newCenter[2].y <= newCenter[0].y) {
        if (newCenter[1].x > newCenter[2].x) {
            index_right = 2;
            index_down = 1;
        }
        else {
            index_right = 1;
            index_down = 2;
        }
    }
    else {
        int x_diff = newCenter[1].x - newCenter[0].x + newCenter[2].x - newCenter[0].x;
        if (x_diff > 0) {
            if (newCenter[1].y > newCenter[2].y) {
                index_right = 2;
                index_down = 1;
            }
            else {
                index_right = 1;
                index_down = 2;
            }
        }
        else {
            if (newCenter[1].y > newCenter[2].y) {
                index_right = 1;
                index_down = 2;
            }
            else {
                index_right = 2;
                index_down = 1;
            }
        }
    }

    
    cv::Mat drawing_three = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
    for (int i = 0;i < newContours.size();i++) {
            cv::drawContours( drawing_three, newContours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }
    // cv::imshow("three contours",drawing_three);
    vector<vector<cv::Point> > approxContours(3);
    for (int i = 0;i < newContours.size();i++) {
        cv::approxPolyDP(newContours[i],approxContours[i],5,true);
    }

    cv::Mat approx = cv::Mat::zeros(imageAfterResize.size(),CV_8U);
    for (int i = 0;i < newContours.size();i++) {
            cv::drawContours( approx, approxContours, i ,cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point() );
    }

    std::vector<cv::Point2f> corners;
	int max_corners = 12;
	double quality_level = 0.01;
	double min_distance = 10.0;
	int block_size = 3;
	bool use_harris = false;
	double k = 0.04;
    cv::Mat drawingCorners = cv::Mat::zeros(approx.size(),CV_8U);
	//角点检测
	cv::goodFeaturesToTrack(approx, 
							corners, 
							max_corners, 
							quality_level, 
							min_distance, 
							cv::Mat(), 
							block_size, 
							use_harris, 
							k);
 
	
    
    vector<cv::Point2f> cen;
    vector<cv::Point2f> down;
    vector<cv::Point2f> right;
    for (int i = 0;i < corners.size();i++) {
        double to_cen = caculateDis(newCenter[index_con],corners[i]);
        double to_down = caculateDis(newCenter[index_down],corners[i]);
        double to_right = caculateDis(newCenter[index_right],corners[i]);

        if (to_cen < to_down && to_cen < to_right)
            cen.push_back(corners[i]);
        else if (to_down < to_cen && to_down < to_right)
            down.push_back(corners[i]);
        else if (to_right < to_cen && to_right < to_down)
            right.push_back(corners[i]);
    }
    // //将检测到的角点绘制到原图上
	// for (int i = 0; i < right.size(); i++)
	// {
	// 	cv::circle(drawingCorners, right[i], 1, cv::Scalar(255), 2, 8, 0);
	// }
    // cv::imshow("drawingCorners",drawingCorners);

    //用两条直线相交的方式算第四个定位点
    cv::Point2f down_left;
    cv::Point2f down_right;
    cv::Point2f right_up;
    cv::Point2f right_down;
    cv::Point2f cen_right;
    cv::Point2f cen_left;
    
    double max_1 = 0;
    double index_max1 = -1;
    double max_2 = 0;
    double index_max2 = -1;

    for (int i = 0;i < approxContours[index_down].size();i++) {
        double tmp = caculateDis(approxContours[index_down][i],newCenter[index_con]);
        if (tmp >= max_1) {
            max_2 = max_1;
            index_max2 = index_max1;
            max_1 = tmp;
            index_max1 = i;
        }
        else if (tmp >= max_2) {
            max_2 = tmp;
            index_max2 = i;
        }
    }
    if (caculateDis(approxContours[index_down][index_max1],newCenter[index_right]) > caculateDis(approxContours[index_down][index_max2],newCenter[index_right])) {
        down_left = approxContours[index_down][index_max1];
        down_right = approxContours[index_down][index_max2];
    }
    else {
        down_left = approxContours[index_down][index_max2];
        down_right = approxContours[index_down][index_max1];
    }
    // cout <<"down_left"<<down_left.x<< " " << down_left.y<<endl;
    // cout <<"down_right"<<down_right.x<< " "<<down_right.y<<endl;
    // cv::circle(drawing_three, down_right, 1, cv::Scalar(255), 2, 8, 0);
    // cv::imshow("drawingCorners",drawing_three);

    max_1 = 0;
    index_max1 = -1;
    max_2 = 0;
    index_max2 = -1;

    for (int i = 0;i < approxContours[index_right].size();i++) {
        double tmp = caculateDis(approxContours[index_right][i],newCenter[index_con]);
        if (tmp >= max_1) {
            max_2 = max_1;
            index_max2 = index_max1;
            max_1 = tmp;
            index_max1 = i;
        }
        else if (tmp >= max_2) {
            max_2 = tmp;
            index_max2 = i;
        }
    }
    if (caculateDis(approxContours[index_right][index_max1],newCenter[index_down]) > caculateDis(approxContours[index_right][index_max2],newCenter[index_down])) {
        right_up = approxContours[index_right][index_max1];
        right_down = approxContours[index_right][index_max2];
    }
    else {
        right_up = approxContours[index_right][index_max2];
        right_down = approxContours[index_right][index_max1];
    }
    // cv::circle(drawing_three, right_down, 1, cv::Scalar(255), 2, 8, 0);
    // cv::imshow("drawingCorners",drawing_three);

    max_1 = 0;
    index_max1 = -1;
    max_2 = 0;
    index_max2 = -1;


    for (int i = 0;i < approxContours[index_con].size();i++) {
        double tmp = caculateDis(approxContours[index_con][i],newCenter[index_down]);
        if (tmp >= max_1) {
            max_2 = max_1;
            index_max2 = index_max1;
            max_1 = tmp;
            index_max1 = i;
        }
        else if (tmp >= max_2) {
            max_2 = tmp;
            index_max2 = i;
        }
    }
    if (caculateDis(approxContours[index_con][index_max1],newCenter[index_right]) > caculateDis(approxContours[index_con][index_max2],newCenter[index_right])) {
        cen_left = approxContours[index_con][index_max1];
        cen_right = approxContours[index_con][index_max2];
    }
    else {
        cen_left = approxContours[index_con][index_max2];
        cen_right = approxContours[index_con][index_max1];
    }

    
    

    vector<cv::Point2f> down_line_points;
    for (int i = 0;i < newContours[index_down].size();i++) {
        int flag = 0;
        if(newContours[index_down][i].x == down_left.x && newContours[index_down][i].y == down_left.y) {
            for (int j = i;;j++) {
                
                down_line_points.push_back(newContours[index_down][j]);
                if (newContours[index_down][j].x == down_right.x && newContours[index_down][j].y == down_right.y) {
                    break;
                    flag = 1;
                }
                
            }
        }
        if (flag)
            break;
    }
    
    cv::Vec4f downline;  //(cos, sin, x0,y0)
    cv::fitLine(cv::Mat(down_line_points), downline, CV_DIST_L2, 0, 0.01, 0.01);


    	
    double down_x0 = downline[2];
    double down_y0 = downline[3];
    double down_x1 = down_x0 + 200 * downline[0];
    double down_y1 = down_y0 + 200 * downline[1];


    // cv::line(drawing_three, cv::Point(down_x0, down_y0), cv::Point(down_x1, down_y1), cv::Scalar(255), 2);

    // cv::imshow("test",drawing_three);

    double a1,b1,c1;
    double k1;
    double b_1;
    if (down_x0 != down_x1) {
        k1 = (down_y0 - down_y1) / (down_x0 - down_x1);
        b_1 = down_y0 - k1 * down_x0;
        a1 = k1;
        b1 = -1;
        c1 = b_1;
    }
    else {
        a1 = 1;
        b1 = 0;
        c1 = -down_x0;
    }
    // cout <<"first equation:"<<endl;
    // cout <<a1<<" "<<b1<<" "<<c1<<endl;

    vector<cv::Point2f> right_line_points;
    for (int i = 0;i < newContours[index_right].size();i++) {
        int flag = 0;
        if(newContours[index_right][i].x == right_down.x && newContours[index_right][i].y == right_down.y) {
            for (int j = i;;j++) {
                
                right_line_points.push_back(newContours[index_right][j]);
                if (newContours[index_right][j].x == right_up.x && newContours[index_right][j].y == right_up.y) {
                    break;
                    flag = 1;
                }
                
            }
        }
        if (flag)
            break;
    }
    
    cv::Vec4f rightline;  //(cos, sin, x0,y0)
    cv::fitLine(cv::Mat(right_line_points), rightline, CV_DIST_L2, 0, 0.01, 0.01);


    	
    double right_x0 = rightline[2];
    double right_y0 = rightline[3];
    double right_x1 = right_x0 + 200 * rightline[0];
    double right_y1 = right_y0 + 200 * rightline[1];

    // cv::line(drawing_three, cv::Point(right_x0, right_y0), cv::Point(right_x1, right_y1), cv::Scalar(255), 2);

    // cv::imshow("test",drawing_three);
   

    double a2,b2,c2;
    double k2;
    double b_2;
    if (right_x0 != right_x1) {
        k2 = (right_y0 - right_y1) / (right_x0 - right_x1);
        b_2 = right_y0 - k2 * right_x0;
        a2 = k2;
        b2 = -1;
        c2 = b_2;
    }
    else {
        a2 = 1;
        b2 = 0;
        c2 = -right_x0;
    }
    // cout <<"Second equation"<<endl;
    // cout <<a2<<" "<<b2<<" "<<c2<<endl;


    
    cv::Point2f fourth;
    fourth.y = -(c1 * a2 - c2 * a1) / (b1 * a2 - a1 * b2);
    fourth.x = -(c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1);




    cv::Mat rr = imageAfterResize.clone();

    vector<cv::Point2f> fourCorner;
    fourCorner.push_back(cen_left);
    fourCorner.push_back(down_left);
    fourCorner.push_back(fourth);
    fourCorner.push_back(right_up);
    
    for (int j = 0; j < 4; j++)
	    cv::line(rr, fourCorner[j], fourCorner[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
    resImage = rr.clone();
    //cv::imshow("rr",rr);

    cv::Point2f srcOuad[] = {
        cen_left,
        down_left,
        fourth,
        right_up
    };

    int dst_rows = 900;
    int dst_cols = 900;
    int margin = 0;

    cv::Point2f dstOuad[] = {
        cv::Point2f(margin,margin),
        cv::Point2f(margin,dst_rows-margin),
        cv::Point2f(dst_cols-margin,dst_rows-margin),
        cv::Point2f(dst_cols-margin,margin)
    };

    
    cv::Mat wrapMat = cv::getPerspectiveTransform(srcOuad,dstOuad);

    cv::Mat A(8,8,CV_32FC1);
    cv::Mat b(8,1,CV_32FC1);

    for (int i = 0,j = 0;i < 4;i++,j=j+2) {
        A.at<float>(j,0) = srcOuad[i].x;
        A.at<float>(j,1) = srcOuad[i].y;
        A.at<float>(j,2) = 1;
        A.at<float>(j,3) = 0;
        A.at<float>(j,4) = 0;
        A.at<float>(j,5) = 0;
        A.at<float>(j,6) = -srcOuad[i].x * dstOuad[i].x;
        A.at<float>(j,7) = -srcOuad[i].y * dstOuad[i].x;

        A.at<float>(j+1,0) = 0;
        A.at<float>(j+1,1) = 0;
        A.at<float>(j+1,2) = 0;
        A.at<float>(j+1,3) = srcOuad[i].x;
        A.at<float>(j+1,4) = srcOuad[i].y;
        A.at<float>(j+1,5) = 1;
        A.at<float>(j+1,6) = -srcOuad[i].x * dstOuad[i].y;
        A.at<float>(j+1,7) = -srcOuad[i].y * dstOuad[i].y;

        b.at<float>(j,0) = dstOuad[i].x;
        b.at<float>(j+1,0) = dstOuad[i].y;
    }


    cv::Mat M = A.inv() * b;
    cv::Mat MyWrapMat(3,3,CV_32FC1);
    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            if (i == 2 && j== 2) {
                MyWrapMat.at<float>(i,j) = 1;
            }
            else {
                MyWrapMat.at<float>(i,j) = M.at<float>(i*3 + j,0);
            }
        }
    }

    cv::Mat MyWrapMat_inv = MyWrapMat.inv();
    cv::Mat MyDst(dst_rows,dst_cols,CV_8U);

    // cv::imshow("grey",greyImage);
    // for (int i = 0;i < MyDst.rows;i++) {
    //     for (int j = 0;j < MyDst.cols;j++) {
            
    //         cv::Mat pixel(3,1,CV_32FC1);
    //         pixel.at<float>(0,0) = i;
    //         pixel.at<float>(1,0) = j;
    //         pixel.at<float>(2,0) = 1;
    //         cv::Mat srcPixel = MyWrapMat_inv * pixel;
    //         double s = 1 / srcPixel.at<float>(2,0);
    //         double src_x = srcPixel.at<float>(0,0) * s;
    //         double src_y = srcPixel.at<float>(1,0) * s;
    //         int src_lowx = src_x;
    //         int src_lowy = src_y;
    //         int src_highx = src_x + 1;
    //         int src_highy = src_y + 1;
    //         double x_direction_low = greyImage.at<uchar>(src_lowy,src_lowx) * (src_x-src_lowx) +
    //                                 greyImage.at<uchar>(src_highy,src_lowx) * (src_highx - src_x);

    //         double x_direction_high = greyImage.at<uchar>(src_lowy,src_highx) * (src_x-src_lowx) +
    //                                 greyImage.at<uchar>(src_highy,src_highx) * (src_highx - src_x);

    //         double goal = x_direction_low * (src_y - src_lowy) + x_direction_high * (src_highy - src_y);
    //         MyDst.at<uchar>(i,j) = goal; 
    //     }
       
    // }
    // cv::threshold(MyDst, MyDst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // cv::imshow("My Result",MyDst);

    cv::Mat image_grey(imageAfterResize.size(),CV_8U);

    cv::cvtColor(imageAfterResize,image_grey,CV_RGB2GRAY);

    cv::Mat final_dst(dst_rows,dst_cols,CV_8U);
    cv::warpPerspective(image_grey,final_dst,MyWrapMat,final_dst.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar());



    int block = final_dst.cols / 6;
    if (block % 2 == 0)
        block += 1;
    myAdaptiveThreshold(final_dst,final_dst, 255,cv::THRESH_BINARY, block, -2);
    //cv::threshold(final_dst, final_dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //cv::imshow("perspective",final_dst);
    
    PerspectiveImage = final_dst.clone();
  


    //准备解码
    float m11 = MyWrapMat.at<float>(0,0);
    float m12 = MyWrapMat.at<float>(0,1);
    float m13 = MyWrapMat.at<float>(0,2);
    float m21 = MyWrapMat.at<float>(1,0);
    float m22 = MyWrapMat.at<float>(1,1);
    float m23 = MyWrapMat.at<float>(1,2);
    float m31 = MyWrapMat.at<float>(2,0);
    float m32 = MyWrapMat.at<float>(2,1);
    cv::Point2f down_left_Perspective;
    down_left_Perspective.x = (m11 * down_left.x + m12 * down_left.y + m13) / (m31 * down_left.x + m32 * down_left.y + 1);
    down_left_Perspective.y = (m21 * down_left.x + m22 * down_left.y + m23) / (m31 * down_left.x + m32 * down_left.y + 1);

    cv::Point2f down_right_Perspective;
    down_right_Perspective.x = (m11 * down_right.x + m12 * down_right.y + m13) / (m31 * down_right.x + m32 * down_right.y + 1);
    down_right_Perspective.y = (m21 * down_right.x + m22 * down_right.y + m23) / (m31 * down_right.x + m32 * down_right.y + 1);


    cv::Point2f cen_left_Perspective;
    cen_left_Perspective.x = (m11 * cen_left.x + m12 * cen_left.y + m13) / (m31 * cen_left.x + m32 * cen_left.y + 1);
    cen_left_Perspective.y = (m21 * cen_left.x + m22 * cen_left.y + m23) / (m31 * cen_left.x + m32 * cen_left.y + 1);

    cv::Point2f cen_right_Perspective;
    cen_right_Perspective.x = (m11 * cen_right.x + m12 * cen_right.y + m13) / (m31 * cen_right.x + m32 * cen_right.y + 1);
    cen_right_Perspective.y = (m21 * cen_right.x + m22 * cen_right.y + m23) / (m31 * cen_right.x + m32 * cen_right.y + 1);
    
    
    double cell = (down_right_Perspective.x - down_left_Perspective.x + cen_right_Perspective.x - cen_left_Perspective.x) / 14;
    
    int code_rows = final_dst.rows / cell ;
    int code_cols = final_dst.cols / cell ;
 
    
    int code[code_rows + 1][code_cols + 1];
    for (int i = 0;i < code_rows;i++) {
        for (int j = 0;j < code_cols;j++) {
            code[i][j] = 0;
        }
    }
    int ii = 0;
    int end_flag = 0;
    for(int i = 0;i < final_dst.rows;i = ii * cell) {
        int j = 0;
        int end = final_dst.cols;

        int jj = 0;
        int k;
        for (;j < end;j = jj * cell) {
            int num = 0;
            
            for(k = i;k < i + cell && k < final_dst.rows;k++) {
                for (int l = j;l < j + cell && l < final_dst.cols;l++) {
                    if (final_dst.at<uchar>(k,l) == 0)
                        num--;
                    else
                        num++;
                }
            }
            if (num >= 0)
                code[ii][jj] = 1;
            else
                code[ii][jj] = 0;


            jj++;
        }
        ii++;
        
        if (k + cell / 2 < i + cell) {
            end_flag = 1;
          
        }
    }
    
    int j_end = 0;
    // cout <<end_flag<<endl;
    if (end_flag) {
        j_end = code_cols + 1;
    }
    else {
        j_end = code_cols;
    }
    // for (int i = 0;i < code_rows + 1;i++) {
    //     for (int j = 0;j < j_end;j++) {
    //         cout <<code[i][j]<<" ";
    //     }
    //     cout <<endl;
    // }
    // cout <<endl<<endl;
      
    for (int i = 0;i < code_rows + 1;i++) {
        vector<int> tmp;
        for (int j = 0;j < j_end;j++) {
            tmp.push_back(code[i][j]);
        }
        resCode.push_back(tmp);
    }
    
   
}

