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
		filename = cv::String(".\..\liantong.jpg");
	}
	ImageTraversal(cv::String& _filename) : isOpened(false), filename(_filename) {
	}
	~ImageTraversal(){
	}
	void togray() {
		if (!isOpened) open();
		cvtColor(img, img, CV_BGR2GRAY);
		img.copyTo(img_backup);
	}
	bool open(void) {
		img = cv::imread(filename, IMREAD_COLOR);
		if (img.empty()) {
			std::cout << "Could not open or find the image" << std::endl;
			isOpened = false;
			return false;
		}
		img_backup = img.clone();
		isOpened = true;
		return true;
	}

	cv::Mat copy(void) {
		return img.clone();
	}

	bool recover(void) {
		if (isOpened) {
			img = img_backup.clone();
			return true;
		}
		return false;
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
	大津法：
		大津法就是最大类间方差，通过统计图像直方图信息来自动确定阈值，从而区分前景与背景
	算法流程：
		1. 遍历图像像素，统计每个像素值出现的次数，即255个bin，统计每个bin像素的个数；
		2. 遍历0到255像素值，以i像素值为区分当前分类的阈值，在（0-i)范围内计算前景像素
		平均灰度u0, 前景部分像素点所占比例w0,在（i~255）计算背景像素背景像素平均灰度u1，
		背景部分像素点所占比例w1;
		3. 计算当前类间方差 varValue = w0*w1*(u1-u0)*(u1-u0);
		4. 选择最大类间方差varValue时的i像素阈值作为区分前景与背景的最佳阈值；
	*/
	int Ostu_threshold() {
		int threshold = 0;
		double maxVarValue = 0; //
		double w0 = 0, w1 = 0;  //背景与前景像素点所占比例
		double u0 = 0, u1 = 0;  //背景与前景像素点平均灰度
		double histgram[256] = { 0 }; 
		double numPixels = static_cast<double>(img.rows) * img.cols;

		//统计256bin中各自的像素个数
		for (int i = 0; i < img.rows; i++) {
			const uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				histgram[rowData[j]]++;
			}
		}

		//前景像素统计
		for (int i = 0; i < 255; i++) {
			w0 = 0, w1 = 0; 
			u0 = 0, u1 = 0;
			for (int j = 0; j <= i; j++) {
				w0 += histgram[j]; //以i为阈值，统计前景像素的个数
				u0 += j * histgram[j]; //以i为阈值，计算前景像素的总灰度和
			}
			u0 /= w0; //前景像素平均灰度
			w0 /= numPixels; //前景像素所占比例
			

			for (int j = i + 1; j <= 255; j++) {
				w1 += histgram[j]; //背景像素个数
				u1 += j * histgram[j]; //背景像素灰度总和
			}
			u1 /= w1; //背景像素平均灰度
			w1 /= numPixels; // 平均像素所占均值

			double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //当前类间方差
			if (maxVarValue < variance) {
				maxVarValue = variance;
				threshold = i;
			}
		}
		return threshold;
	}

	int Ostu_threshold_v2() {
		int threshold = 0;
		double maxVarValue = 0; //
		double w0 = 0, w1 = 0;  //背景与前景像素点所占比例
		double u0 = 0, u1 = 0;  //背景与前景像素点平均灰度
		double histgram[256] = { 0 };
		double numPixels = static_cast<double>(img.rows) * img.cols;

		//
		double allPixelsGray = 0;
		//统计256bin中各自的像素个数
		for (int i = 0; i < img.rows; i++) {
			const uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				histgram[rowData[j]]++;
				allPixelsGray += rowData[j];
			}
		}

		//前景像素统计
		for (int i = 0; i < 255; i++) {
			w0 = 0, w1 = 0;
			u0 = 0, u1 = 0;
			for (int j = 0; j <= i; j++) {
				w0 += histgram[j]; //以i为阈值，统计前景像素的个数
				u0 += j * histgram[j]; //以i为阈值，计算前景像素的总灰度和
			}
			
			u1 = (allPixelsGray - u0) / (numPixels - w0); //背景像素平均灰度
			u0 /= w0; //前景像素平均灰度
			w0 /= numPixels; //前景像素所占比例
			w1 = 1 - w0; // 平均像素所占比例
			

			double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //当前类间方差
			if (maxVarValue < variance) {
				maxVarValue = variance;
				threshold = i;
			}
		}
		return threshold;
	}


	int Ostu_threshold_v3() {
		int threshold = 0;
		double maxVarValue = 0; //
		double w0 = 0, w1 = 0;  //背景与前景像素点所占比例
		double u0 = 0, u1 = 0;  //背景与前景像素点平均灰度
		double histgram[256] = { 0 };
		double numPixels = static_cast<double>(img.rows) * img.cols;

		//
		double allPixelsGray = 0;
		//统计256bin中各自的像素个数
		for (int i = 0; i < img.rows; i++) {
			const uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				histgram[rowData[j]]++;
				allPixelsGray += rowData[j];
			}
		}

		//前景像素统计
		double w0_accum = 0;
		double u0_accum = 0;
		for (int i = 0; i < 255; i++) {
			w0_accum += histgram[i]; //以i为阈值，统计前景像素的个数
			u0_accum += i * histgram[i]; //以i为阈值，计算前景像素的总灰度和
 
			w0 = w0_accum; 
			u0 = u0_accum;
			u1 = (allPixelsGray - u0) / (numPixels - w0); //背景像素平均灰度
			u0 /= w0; //前景像素平均灰度
			w0 /= numPixels; //前景像素所占比例
			w1 = 1 - w0; // 平均像素所占比例


			double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //当前类间方差
			if (maxVarValue < variance) {
				maxVarValue = variance;
				threshold = i;
			}
		}
		return threshold;
	}



	/*
	二值化
	*/
	void binarize(double threshold) {
		for (int i = 0; i < img.rows; i++) {
			uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				if (rowData[j] <= threshold)
					rowData[j] = 0;
				else
					rowData[j] = 255;
			}
		}
	}

	//void Ostu_opencv() {
	//	cv::Mat _img;
	//	threshold(img, _img, 150, 255, THRESH_BINARY | THRESH_OTSU);
	//	img = _img;
	//}

private:
	bool isOpened;
	string filename;
	cv::Mat img;
	cv::Mat img_backup;
};