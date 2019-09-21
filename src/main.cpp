#include "qrcode.h"
using namespace std;


int main(int argc, char const *argv[])
{
    MyQRCodeLocating* test = new MyQRCodeLocating();
    bool isVideo = false;
    int num = 0;
    int image_numbers = 39;
    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat resImage;
    while (true)
    {
        string resFilename;
        if (isVideo) {
            cap >> image;

        }
        else {
            stringstream ss;
            ss << num;
            string str = ss.str();
            string filename = "../src_image/" + str + ".jpg";
            resFilename = "../result_image/" + str + ".jpg";
            image = cv::imread(filename);
        }

        if (image.empty()) {
            cout << "No image" <<endl;
            break;
        }
       
        
        test -> imageProcess(image,resImage);
        cv::imshow("result",resImage);
        if (!isVideo)
            cv::imwrite(resFilename,resImage);
        char c;
        if (isVideo)
            c = cv::waitKey(1);
        else
            c = cv::waitKey(0);

        if(c == 'q' || c == 'Q')
            break;
        else {
            if (c == 'p')
                num--;
            else
                num++;
            if (num < 0)
                num = image_numbers - 1;
            
            num = num % image_numbers;
            continue;
        }
        
    }
    
    return 0;
}
