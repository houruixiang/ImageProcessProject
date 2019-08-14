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
	int size() { return img.rows * img.cols; }
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



	void detectHScolor(double minHue, double maxHue, double minSat, double maxSat, cv::Mat& mask) {
		cv::Mat hsv;
		cv::cvtColor(img, hsv, CV_BGR2HSV);
		
		std::vector<cv::Mat> channels;
		cv::split(hsv, channels);
		// 色调掩码
		cv::Mat mask1;
		cv::threshold(channels[0], mask1, maxHue, 255, cv::THRESH_BINARY_INV); // 小于maxHue
		cv::Mat mask2;
		cv::threshold(channels[0], mask2, minHue, 255, cv::THRESH_BINARY); // 大于minHue
		cv::Mat hueMask;
		if (minHue < maxHue) 
			hueMask = mask1 & mask2;
		else 
			hueMask = mask1 | mask2; //区间穿越0度中轴线

		// 饱和度掩码
		cv::Mat satMask;
		cv::inRange(channels[1], minSat, maxSat, satMask);

		// 组合掩码
		mask = hueMask & satMask;

	}


private:
	bool isOpened;
	string filename;
	cv::Mat img;
};