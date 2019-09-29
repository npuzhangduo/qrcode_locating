#include "qrcode.h"
using namespace std;


int main(int argc, char const *argv[])
{
    
    bool isVideo = false;
    int num = 0;
    int image_numbers = 39;
    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat Location_resImage;
    cv::Mat PerspectiveImage;
    vector<vector<int> >Code;
    while (true)
    {
        MyQRCode* test = new MyQRCode();
        string resFilename;
        string DecodeFilename;

        if (isVideo) {
            cap >> image;

        }
        else {
            stringstream ss;
            ss << num;
            string str = ss.str();
            string filename = "../src_image/" + str + ".jpg";
            resFilename = "../Location_resImage/" + str + ".jpg";
            DecodeFilename = "../Decode_result/" + str + ".txt";
            image = cv::imread(filename);
        }

        if (image.empty()) {
            cout << "No image" <<endl;
            break;
        }
       
        
        bool find = test -> imageProcess(image,Location_resImage,PerspectiveImage,Code);
    
        cv::imshow("Location_resImage",Location_resImage);
        cv::imshow("PerspectiveImage",PerspectiveImage);

        if (!isVideo)
            cv::imwrite(resFilename,Location_resImage);

        if (find) {
            ofstream out;
            out.open(DecodeFilename);
            for (int i = 0;i < Code.size();i++) {
                for (int j = 0;j < Code[i].size();j++) {
                    out <<Code[i][j]<<" ";
                }
                out <<endl;
            }
            out.close();
        }




        char c;
        if (isVideo)
            c = cv::waitKey(1);
        else
            c = cv::waitKey(0);

        delete test;
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
