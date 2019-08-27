#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;


class HarrisDetector {
private:
	//32位浮点的角点强度图像
	cv::Mat cornerStrength;
	//32位浮点的阈值化图像
	cv::Mat cornerTh;
	//局部最大值图像
	cv::Mat localMax;
	//平滑偏导数的邻域尺度
	int neighborhood;
	//梯度计算的口径
	int aperture;
	//Harris参数
	double k;
	//阈值计算的最大强度
	double maxStrength;
	//计算得到的阈值
	double threshold;
	//非最大值抑制的邻域尺寸
	int nonMaxSize;
	//非最大值抑制的内核
	int nonMaxKernel;
public:
	HarrisDetector() : neighborhood(3), aperture(3),
		k(0.01), maxStrength(0.0),
		threshold(0.01), nonMaxSize(3) {

	}

	// 角点检测
	void detect(const cv::Mat& image) {
		//计算Harris
		cv::cornerHarris(image, cornerStrength,
			neighborhood, aperture, k);
		//计算内部阈值最大值
		cv::minMaxLoc(cornerStrength, 0, &maxStrength);

		//非极大值抑制，作用是排除掉紧邻的Harris角点，即Harris角点不仅要有
		//高于指定阈值的得分，还必须是局部范围内的最大值
		//膨胀运算会在邻域内把每个像素都替换成最大值，只有局部最大值是不变的，
		//localMax矩阵只有在局部最大值位置上才是真，
		cv::Mat dilated; //膨胀图像
		cv::dilate(cornerStrength, dilated, cv::Mat());
		cv::compare(cornerStrength, dilated, localMax, cv::CMP_EQ);

	}

	//用Harris值得到角点分布图
	cv::Mat getCornerMap(double qualityLevel) {
		cv::Mat cornerMap;
		threshold = qualityLevel * maxStrength;
		cv::threshold(cornerStrength, cornerTh, threshold, 255, cv::THRESH_BINARY); //二值化

		//转换成8bit图像
		cornerTh.convertTo(cornerMap, CV_8U);

		//非极大值抑制
		cv::bitwise_and(cornerMap, localMax, cornerMap);

		return cornerMap;
	}

	//坐标体系中的零点坐标为图片的左上角，X轴为图像矩形的上面那条水平线；Y轴为图像矩形左边的那条垂直线。该坐标体系在诸如结构体Mat, Rect, Point中都是适用的。
	//虽然网上有学着说OpenCV中有些数据结构的坐标原点是在图片的左下角，但是我暂时还没碰到过）。

	//在使用image.at(x1, x2)来访问图像中点的值的时候，x1并不是图片中对应点的x轴坐标，而是图片中对应点的y坐标。
	//因此其访问的结果其实是访问image图像中的Point(x2, x1)点，即与image.at(Point(x2, x1))效果相同。

	//如果所画图像是多通道的，比如说image图像的通道数时n，则使用Mat::at(x, y)时，其x的范围依旧是0到image的height，
	//而y的取值范围则是0到image的width乘以n，因为这个时候是有n个通道，所以每个像素需要占有n列。但是如果在同样的情况下，
	//使用Mat::at(point)来访问的话，则这时候可以不用考虑通道的个数，因为你要赋值给获取Mat::at(point)的值时，
	//都不是一个数字，而是一个对应的n维向量。
	//用角点分布图得到特征点
	void getCorners(std::vector<cv::Point>& points, const cv::Mat& cornerMap) {
		for (int y = 0; y < cornerMap.rows; y++) {
			const uchar* rowPtr = cornerMap.ptr<uchar>(y);
			for (int x = 0; x < cornerMap.cols; x++) {
				if (rowPtr[x]) {
					points.push_back({x, y});
				}
			}
		}
	}

	//获得特征点
	void getCorners(std::vector<cv::Point>& points, double qualityLevel) {
		//获得角点分布图
		cv::Mat cornerMap = getCornerMap(qualityLevel);
		//获得角点
		getCorners(points, cornerMap);
	}

	//在特征点位置画圆形
	void drawOnImage(cv::Mat& image, const std::vector<cv::Point>& points,
		cv::Scalar color = cv::Scalar(255, 255, 255),
		int raduis = 3, int thickness = 1) {
		int lens = points.size();
		//对所有角点
		for(int i=0; i<lens; i++)
			cv::circle(image, points[i], raduis, color, thickness);
	}
	 
};


void HarrisDetector_all(cv::Mat& image) {
	HarrisDetector harris;
	//计算harris值
	harris.detect(image);
	//检测Harris角点
	std::vector<cv::Point>points;
	harris.getCorners(points, 0.03);
	harris.drawOnImage(image, points);
}












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



private:
	bool isOpened;
	string filename;
	cv::Mat img;
};