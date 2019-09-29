# 目录结构说明：
src_image 文件夹为测试图片的原图片\
Location_resImage 文件夹为定位结果图片 \
Decode_result 文件夹保存为解码结果txt文件，以矩阵形式保存\
src 文件夹中为源代码 \
include 文件夹中为头文件 \
CMakeLists.txt文件 \
configure.sh为编译该工程的自动化脚本 \
clean.sh为清理编译生成的build文件夹的脚本 

# 类接口说明：
MyQRCode类对外开放一个接口：

bool imageProcess(cv::Mat &src,cv::Mat &res,cv::Mat &PersRes,vector<vector<int> > &Code);

第一个参数src为源图片\
第二个参数二维码定位的结果，用红框框住了二维码\
第三个参数为使用透视变换进行标准化后的二维码图片\
第四个参数为解码的结果，为一个二维数组



# 工程的编译以及运行方法：
首先在终端运行configure.sh完成工程的编译，
完成编译后可执行文件main变出现在build文件夹中，
然后进去build文件夹，即可在终端执行该工程。


代码测试方法，在终端运行可执行文件后，便可以按下键盘上的n键测试下一张图片，也可以按下p键回到上一张图片。
若要新增测试图片，请将新增的图片放在src_image中，保证格式为.jpg格式，同时序号依次加1,然后在main.cpp中将image_numbers改成对应图片的数量，然后重新编译即可。

代码还有另一种模式，在main.cpp中将main函数中的bool变量isVideo设置为true，便可以调用电脑自带的摄像头，实时定位摄像头图像中的二维码。



