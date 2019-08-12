#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include <random>
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
	bool recover(void) {
		if (isOpened) {
			img = img_backup.clone();
			return true;
		}
		return false;
	}
	bool open(void) {
		img = cv::imread(filename, IMREAD_GRAYSCALE);
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


	// mean filter
	cv::Vec3b meanFilter_vec(std::vector<cv::Mat>& channels, int y, int x, int raduis, int size) {
		uchar x0 = meanFilter_scalar(channels[0], y, x, raduis, size);
		uchar x1 = meanFilter_scalar(channels[1], y, x, raduis, size);
		uchar x2 = meanFilter_scalar(channels[2], y, x, raduis, size);
		return cv::Vec3b(x0, x1, x2);
	}
	int meanFilter_scalar(cv::Mat& _img, int y, int x, int raduis, int size) {
		int sum = 0;
		for (int j = -raduis; j <= raduis; j++) {
			uchar* rowData = _img.ptr<uchar>(j + y);
			for (int i = -raduis; i <= raduis; i++) {
				sum += rowData[i + x];
			}
		}
		return sum / size;
	}

	void meanFilter(cv::Mat& _img, int kernelSize) {
		int raduis = kernelSize / 2;
		int size = kernelSize * kernelSize;
		//拓展边界
		copyMakeBorder(img, _img, raduis, raduis, raduis, raduis, cv::BORDER_REFLECT);
		int rows = _img.rows;
		int cols = _img.cols;
 
		for (int y = raduis; y < rows - raduis; y++) {
			uchar* rowData = img.ptr<uchar>(y - raduis);
			for (int x = raduis; x < cols - raduis; x++) {
				rowData[x - raduis] = meanFilter_scalar(_img, y, x, raduis, size);
			}
		}
		//std::vector<cv::Mat> channels(3);
		//split(_img, channels);
		//for (int y = raduis; y < rows - raduis; y++) {
		//	cv::Vec3b* rowData = img.ptr<cv::Vec3b>(y - raduis);
		//	for (int x = raduis; x < cols - raduis; x++) {
		//		rowData[x - raduis] = meanFilter_scalar(_img, y, x, raduis, size);
		//	}
		//}
	}
	void meanFilter_blur(int kernelSize) {
		cv::blur(img, img, Size(kernelSize, kernelSize));
	}



	//-------------------------------------------------------------------------
	// Guass Filter
	void addGaussNoise(int mu, int sigma) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> d(mu, sigma);

		int rows = img.rows;
		int cols = img.cols * img.channels();
		for (int y = 0; y < rows; y++) {
			auto rowData = img.ptr<uchar>(y);
			for (int x = 0; x < cols; x++) {
				auto temp =  d(gen) + rowData[x];
				rowData[x] = cv::saturate_cast<uchar>(temp);
			}
		}
	}

	const float pi = 3.1415926;
	vector<vector<float>> gaussTemplate(int size, float sigma) {
		int xcore = size / 2; 
		int ycore = size / 2;
		vector<vector<float>> res;
		for (int x = 0; x < size; x++) {
			vector<float> ans;
			float base = 1.0 / 2 / pi / sigma / sigma;
			for (int y = 0; y < size; y++) {
				float tmp1 = (pow(x - xcore, 2) + pow(y - ycore, 2))/ 2 / sigma / sigma;
				ans.push_back(base * exp(-tmp1));
			}
			res.push_back(ans);
		}
		//系数归一化
		float factorSum = 0.0;
		for (auto& ans : res) for (auto& x : ans) factorSum += x;
		for (auto& ans : res) for (auto& x : ans) x /= factorSum;
		return res;
	}
	
	uchar GaussFilter_scalar(cv::Mat& _img, vector<vector<float>>& factor, int y, int x, int raduis) {
		float sum = 0.0;
		for (int j = -raduis; j <= raduis; j++) {
 			uchar* rowData = _img.ptr<uchar>(y + j);
			for (int i = -raduis; i <= raduis; i++) {
				sum += rowData[x + i] * factor[j + raduis][i + raduis];
			}
		}
		return static_cast<uchar>(sum);
	}
	void GaussFilter(cv::Mat& _img, int kernelSize, int sigma) {
		vector<vector<float>> factor = gaussTemplate(kernelSize, sigma);
		

		int raduis = kernelSize / 2;
		int size = kernelSize * kernelSize;
		//拓展边界
		copyMakeBorder(img, _img, raduis, raduis, raduis, raduis, cv::BORDER_REFLECT);
		int rows = _img.rows;
		int cols = _img.cols;

		for (int y = raduis; y < rows - raduis; y++) {
			uchar* rowData = img.ptr<uchar>(y - raduis);
			for (int x = raduis; x < cols - raduis; x++) {
				rowData[x - raduis] = GaussFilter_scalar(_img, factor, y, x, raduis);
			}
		}
	}


	void GaussFilter_blur(int kernelSize, int sigma) {
		cv::GaussianBlur(img, img, Size(kernelSize, kernelSize), sigma);
	}



private:
	bool isOpened;
	string filename;
	cv::Mat img;
	cv::Mat img_backup;
};