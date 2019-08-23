#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;



class ImageTraversal {
public:
	ImageTraversal():isOpened(false){
		filename = cv::String("../liantong.jpg");
	}
	ImageTraversal(cv::String& _filename) : isOpened(false), filename(_filename) {
	}
	~ImageTraversal(){
	}

	bool open(void) {
		img = cv::imread(filename, IMREAD_COLOR);
		if (img.empty()) {
			std::cout << "Could not open or find the image" << std::endl;
			isOpened = false;
			return false;
		}
	 
		isOpened = true;
		return true;
	}

	cv::Mat copy(void) {
		return img.clone();
	}


	void display_img(void) {
		if (!isOpened)  open();
		namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("Display window", img);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
	}


	// ---------------------------基于拉普拉斯算子的增强--------------------------------------
	/*
	 原始图像减去拉普拉斯图像           拉普拉斯图像
		  0   -1    0              0    1    0
		  -1   5   -1              1   -4    1
		  0   -1    0              0    1    0   
	
	*/

	void sharpen(cv::Mat& _img) {
		int nchannels = img.channels();
		for (int i = 1; i < img.rows-1; i++) {
			const uchar* prev = img.ptr<const uchar>(i - 1);
			const uchar* curr = img.ptr<const uchar>(i);
			const uchar* next = img.ptr<const uchar>(i + 1);
			uchar* out = _img.ptr<uchar>(i);
			for (int j = nchannels; j < nchannels * (img.cols - 1); j++) {
				*out++ = cv::saturate_cast<uchar>(5 * curr[j] - curr[j - nchannels] - 
					curr[j + nchannels] - prev[j] - next[j]);
			}
		}

		_img.row(0).setTo(cv::Scalar(0));
		_img.row(_img.rows - 1).setTo(cv::Scalar(0));
		_img.col(0).setTo(cv::Scalar(0));
		_img.col(_img.cols - 1).setTo(cv::Scalar(0));
	}


	/*
		载入RGB彩色图像
		灰度化、高斯滤波、Canny边缘检测
		查找轮廓，对轮廓进行标号
		基于标记的分水岭算法
		对每个区域进行随机颜色填充
	*/
	cv::Mat autoWaterShed(void){
		cv::Mat imageGray;
		cv::cvtColor(img, imageGray, CV_RGB2GRAY); //灰度转换
		
		GaussianBlur(imageGray, imageGray, Size(5, 5), 2); //高斯滤波
		imshow("Gray Image", imageGray);

	

		//形态学上腐蚀操作 效果不好
		//cv::dilate(imageGray, imageGray, cv::Mat(), Point(-1, -1), 4);
		//imshow("erode Image", imageGray);  

		////形态学上开操作
		//cv::Mat elem5(3, 3, CV_8U, cv::Scalar(1)); //5x5的结构元，元素为1
		//cv::morphologyEx(imageGray, imageGray, cv::MORPH_OPEN, elem5);
		//imshow("close Image", imageGray);

		////距离变换 用于计算图像中每一个非零点距离 离 自己最近的零点的距离
		////第二个mat矩阵参数保存了每一个点离自己最近的零点的距离信息，图像上的点
		////越亮，代表了距离零点的距离越远，简单的阈值比较来细化字符的轮廓
		//cv::Mat imageThin(imageGray.size(), CV_32FC1);
		//cv::distanceTransform(imageGray, imageThin, CV_DIST_L2, 3);
		//imshow("Thin Image", imageThin);

	
		//float maxDis = 0.0;
		//for (int i = 0; i < imageThin.rows; i++) {
		//	float* data = imageThin.ptr<float>(i);
		//	for (int j = 0; j < imageThin.cols; j++) {
		//		if (data[j] > maxDis) maxDis = data[j]; //获取距离变换中的极大值
		//	}
		//}

		//imageGray = cv::Mat::zeros(imageGray.size(), CV_8UC1);
		//for (int i = 0; i < imageThin.rows; i++) {
		//	float* data = imageThin.ptr<float>(i);
		//	for (int j = 0; j < imageThin.cols; j++) {
		//		if (data[j] > maxDis *0.02)
		//			imageGray.at<uchar>(i, j) = 255;
		//	}
		//}
		//imshow("distanceTransform Image", imageGray);


		cv::Canny(imageGray, imageGray, 80, 150);
		imshow("Canny Image", imageGray);  


		////查找轮廓
  //      vector<vector<Point>> contours;    
		//vector<Vec4i> hierarchy;    
		//cv::findContours(imageGray,contours,hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE,Point());
		//Mat imageContours=Mat::zeros(imageGray.size(),CV_8UC1);  //轮廓     
		//Mat marks(imageGray.size(),CV_32S);   //Opencv分水岭第二个矩阵参数  
		//marks=Scalar::all(0);  
		//int index = 0;  
		//int compCount = 0;  
		//for( ; index >= 0; index = hierarchy[index][0], compCount++ )   
		//{  
		//	// 对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点  
		//	drawContours(marks, contours, index, Scalar::all(uchar(compCount + 1)), 1, 8, hierarchy);
		//	drawContours(imageContours,contours,index,Scalar(255),1,8,hierarchy);    
		//}  

		//cv::Mat marksShows;
		//convertScaleAbs(marks, marksShows);
		//imshow("marksShow", marksShows);
		//imshow("轮廓", imageContours);

		return imageGray;
	
	}

private:
	bool isOpened;
	string filename;
	cv::Mat img;
};