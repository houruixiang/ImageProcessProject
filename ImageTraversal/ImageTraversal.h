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

	//at<typename>(i,j)
	void colorReduce_twoRepeatTravel_at(cv::Mat& _img, int div = 64) {
		int rows = img.rows;
		int cols = img.cols;
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				_img.ptr<uchar>(i, j)[0] = _img.ptr<uchar>(i, j)[0] / div * div + div / 2;
				_img.ptr<uchar>(i, j)[1] = _img.ptr<uchar>(i, j)[1] / div * div + div / 2;
				_img.ptr<uchar>(i, j)[2] = _img.ptr<uchar>(i, j)[2] / div * div + div / 2;
			}
		}
	}

	// 用指针来遍历图像
	void colorReduce_twoRepeatTravel_ptr(cv::Mat& _img, int div = 64) {
		int rows = img.rows;
		int cols = img.cols * img.channels();
		for (int i = 0; i < rows; i++) {
			uchar* data = _img.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				data[j] = data[j] / div * div + div / 2;
			}
		}
	}



	// 用迭代器来遍历图像
	void colorReduce_oneRepeatTravel_iterator(cv::Mat& _img, int div = 64) {
		cv::Mat_<cv::Vec3b>::iterator in_it = img.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator in_iend = img.end<cv::Vec3b>();		
		cv::Mat_<cv::Vec3b>::iterator out_it = _img.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator out_iend = _img.end<cv::Vec3b>();

		while (in_it != in_iend) {
			(*out_it)[0] = (*in_it)[0] / div * div + div / 2;
			(*out_it)[1] = (*in_it)[1] / div * div + div / 2;
			(*out_it)[2] = (*in_it)[2] / div * div + div / 2;
			in_it++;  out_it++;
		}
	}


	//at<typename>(i,j)
	void colorReduce_twoRepeatTravel_at_bit(cv::Mat& _img, int div = 64) {
		int rows = img.rows;
		int cols = img.cols;
		int n = static_cast<int>(log(static_cast<double>(div) / log(2.0) + 0.5));
		uchar mask = 0xff << n;
		uchar div2 = div >> 1;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				_img.ptr<uchar>(i, j)[0] &= mask; _img.ptr<uchar>(i, j)[0] += div2;
				_img.ptr<uchar>(i, j)[1] &= mask; _img.ptr<uchar>(i, j)[1] += div2;
				_img.ptr<uchar>(i, j)[2] &= mask; _img.ptr<uchar>(i, j)[2] += div2;
			}
		}
	}

	// 用指针来遍历图像 这个写法最佳
	void colorReduce_twoRepeatTravel_ptr_bit(cv::Mat& _img, int div = 64) {
		int rows = img.rows;
		int cols = img.cols * img.channels();
		int n = static_cast<int>(log(static_cast<double>(div) / log(2.0) + 0.5));
		uchar mask = 0xff << n;
		uchar div2 = div >> 1;
		for (int i = 0; i < rows; i++) {
			uchar* data = _img.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				data[j] &= mask; 
				data[j] += div2;
			}
		}
	}
	// 用指针来遍历图像 这个写法最佳
	void colorReduce_mask_bit(cv::Mat& _img, int div = 64) {
		int rows = img.rows;
		int cols = img.cols * img.channels();
		int n = static_cast<int>(log(static_cast<double>(div) / log(2.0) + 0.5));
		uchar mask = 0xff << n;
		uchar div2 = div >> 1;
		_img = (img & cv::Scalar(mask, mask, mask)) + cv::Scalar(div2, div2, div2);
	}


	// 用迭代器来遍历图像
	void colorReduce_oneRepeatTravel_iterator_bit(cv::Mat& _img, int div = 64) {
		cv::Mat_<cv::Vec3b>::iterator in_it = img.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator in_iend = img.end<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator out_it = _img.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator out_iend = _img.end<cv::Vec3b>();
		int n = static_cast<int>(log(static_cast<double>(div) / log(2.0) + 0.5));
		uchar mask = 0xff << n;
		uchar div2 = div >> 1;
		while (in_it != in_iend) {
			(*out_it)[0] = ((*in_it)[0] & mask) + div2;
			(*out_it)[1] = ((*in_it)[1] & mask) + div2;
			(*out_it)[2] = ((*in_it)[2] & mask) + div2;
			in_it++;  out_it++;
		}
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

	void sharpen2D(cv::Mat& _img) {
		cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
		kernel.at<float>(1, 1) = 5.0;
		kernel.at<float>(0, 1) = -1.0;
		kernel.at<float>(2, 1) = -1.0;
		kernel.at<float>(1, 0) = -1.0;
		kernel.at<float>(1, 2) = -1.0;

		cv::filter2D(img, _img, img.depth(), kernel);
	}



private:
	bool isOpened;
	string filename;
	cv::Mat img;
};