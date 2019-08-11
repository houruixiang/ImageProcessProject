#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;



class ImageTraversal {
public:
	ImageTraversal():isOpened(false), minSize(3), maxSize(7){
		filename = cv::String("../liantong.jpg");
	}
	ImageTraversal(cv::String& _filename) : isOpened(false), filename(_filename), minSize(3), maxSize(7) {
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

	int size() { return img.rows * img.cols; }
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


	//灰度图
	void toGray() {
		if (img.type() == CV_8UC3) {
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
	}
	//添加盐噪声
	void addSaltNoise(int num) {
		//随机数生成器
		std::random_device rd; // seed
		std::mt19937 gen(rd()); // random number engineer
		auto cols = img.cols;
		if (img.channels() == 1) {
			for (int i = 0; i < num; i++) {
				auto row = static_cast<int>(gen() % img.rows);
				auto col = static_cast<int>(gen() % img.cols);
				img.at<uchar>(row, col) = 255;
			}
		}
		else if (img.channels() == 3) {
			for (int i = 0; i < num; i++) {
				auto row = static_cast<int>(gen() % img.rows);
				auto col = static_cast<int>(gen() % img.cols);
				img.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	//添加椒噪声
	void addPepperNoise(int num) {
		//随机数生成器
		std::random_device rd; // seed
		std::mt19937 gen(rd()); // random number engineer
		auto cols = img.cols;
		if (img.channels() == 1) {
			for (int i = 0; i < num; i++) {
				auto row = static_cast<int>(gen() % img.rows);
				auto col = static_cast<int>(gen() % img.cols);
				img.at<uchar>(row, col) = 0;
			}
		}
		else if (img.channels() == 3) {
			for (int i = 0; i < num; i++) {
				auto row = static_cast<int>(gen() % img.rows);
				auto col = static_cast<int>(gen() % img.cols);
				img.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
			}
		}
	}


	/*
	中值滤波的思想就是比较一定领域内的像素值的大小，取出其中值作为这个领域的中心像素新的值。
	假设对一定领域内的所有像素从小到大进行排序，如果存在孤立的噪声点，比如椒盐噪声（椒噪声——较小的灰度值，
	呈现的效果是小黑点；盐噪声——较大的灰度值，呈现的效果是小白点），那么从小到大排序的这个数组中，
	那些孤立的噪声一定会分布在两边（要么很小，要么很大），这样子取出的中值点可以很好地保留像素信息，而滤除了噪声点的影响。 
    中值滤波器受滤波窗口大小影响较大，用于消除噪声和保护图像细节，两者会存在冲突。如果窗口较小，
	则能较好地保护图像中的一些细节信息，但对噪声的过滤效果就会打折扣；反之，如果窗口尺寸较大则会有较好的噪声过滤效果，
	但也会对图像造成一定的模糊效果，从而丢失一部分细节信息。另外，如果在滤波窗口内的噪声点的个数大于整个窗口内像素的个数，
	则中值滤波就不能很好的过滤掉噪声。

	彩色图像一般不能在rbg空间内做中值滤波，因此这里处理的是灰度图像
	*/
	uchar medianFilter(int row, int col, int kernelSize) {
		std::vector<uchar> pixels;
		int radius = kernelSize / 2;
		for (int y = -radius; y <= radius; y++) {
			uchar* rowData = img.ptr<uchar>(row+y);
			for (int x = -radius; x <= radius; x++) {
				pixels.push_back(rowData[col+x]);
			}
		}
		
		sort(pixels.begin(), pixels.end());
		return pixels[pixels.size() / 2];
	}
	// 这个算法效率极其低下
	void medianFilter(int kernelSize) {
		int raduis = kernelSize / 2;
		// 拓展图像的边界
		cv::copyMakeBorder(img, img, raduis, raduis, raduis, raduis, cv::BORDER_REFLECT);
		int rows = img.rows;
		int cols = img.cols;
		for (int y = raduis; y < rows - raduis; y++) {
			for (int x = raduis; x < cols - raduis; x++) {
				img.at<uchar>(y, x) = medianFilter(y, x, kernelSize);
			}
		}
	}
	
	/*
	在噪声密度不是很大的情况下（根据经验，噪声的出现的概率小于0.2），使用中值滤波的效果不错。
	但是当噪声出现的概率比较高时，原来的中值滤波算法就不是很有效了。只有增大滤波器窗口尺寸，尽管会使图像变得模糊。 
	使用自适应中值滤波器的目的就是，根据预设好的条件，动态地改变中值滤波器的窗口尺寸，以同时兼顾去噪声作用和保护细节的效果。 
 	*/
	uchar adaptiveMedianFilter(int row, int col) {
		std::vector<uchar> pixels;
		int raduis = minSize / 2;
		for (int y = -raduis; y <= row + raduis; y++) {
			uchar* rowData = img.ptr<uchar>(row + y);
			for (int x = -raduis; x <= col + raduis; x++) {
				pixels.push_back(rowData[col + y]);
			}
		}
		sort(pixels.begin(), pixels.end());
		auto min = pixels.front();
		auto max = pixels.back();
		auto mid = pixels[pixels.size() / 2];
		auto zxy = img.at<uchar>(row, col);

		if (min < mid && mid < max) { //mid 非椒盐
			if (min < zxy && zxy < max) //hold
				return zxy;
			else
				return mid;
		}
		else {
			minSize += 2;
			if (minSize <= maxSize)
				return adaptiveMedianFilter(row, col);
			else
				return mid;
		}
	}
	void adaptiveMedianFilter() {
		int raduis = maxSize / 2;
		// 拓展图像的边界
		cv::copyMakeBorder(img, img, raduis, raduis, raduis, raduis, cv::BORDER_REFLECT);
		int rows = img.rows;
		int cols = img.cols;
		for (int y = raduis; y < rows - raduis; y++) {
			for (int x = raduis; x < cols - raduis; x++) {
				img.at<uchar>(y, x) = adaptiveMedianFilter(y, x);
			}
		}
	}

private:
	int minSize, maxSize;
	bool isOpened;
	string filename;
	cv::Mat img;
};