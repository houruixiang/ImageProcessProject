#include <iostream>
#include <vector>
#include <string>
#include "ImageTraversal.h"


int main(void) {

	cv::String fpath = "./../../img/lena.jpg";
	ImageTraversal image = ImageTraversal(fpath);
	image.open();
	int64 start;
	double duration;

	cv::Mat _img = image.copy();
	start = cv::getTickCount();
	image.fast_detect_();
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of open(ms): " << duration << endl;
	
	
	
	
	imshow("Display the processed imgage window", _img); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window


	return 0;
}

 

//经百度说出现这种错误是内存问题，要么就是内存木有释放，要么就是两个指针指向同一个内存，一个指针把它释放掉了等等，
//于是我在代码里面找了很久，改了许多遍 也没有解决，最后我发现我把代码中的vector
//
//std::vector<cv::KeyPoint> keypoints1, keypoints2;
//cv::Mat descriptors1, descriptors2;
//std::vector<cv::DMatch> matches;//匹配结果
//std::vector<cv::DMatch> goodMatches;
//
//从函数中拿出来，作为全局变量，问题尽然解决了，其实我都没想明白这是什么原因。
//vector里面的内存不是不需要手动释放的吗，话说当其在函数里面作为局部变量的时候我也尝试过释放它，
//先是clear，后来发现clear不能释放其内存，于是我按照网上说的用swap来手动释放，可是最终还是没用。
//最后误打误撞的，把它作为全局变量，问题才得以解决，不过解决了就好。
//在debug底下程序已经可以运行了，全都通过了，可接下来，我在release底下又遇到错误了，错误类型如下：
//
//
//
//网上说这种我是添加opencv库的时候添加错了，我仔细看了我的additional dependencies的添加，
//在release底下我确实没带d啊，可怎么就是出现了这种错误呢，要是是程序有问题，那怎么debug底下可以运行呢，
//我很纳闷，尝试了很多方法。最后抱着试试的态度，干脆把release和debug底下都添加了带d的库文件，
//奇葩的是问题尽然结局了。对于这我只能表示相当无语啊！