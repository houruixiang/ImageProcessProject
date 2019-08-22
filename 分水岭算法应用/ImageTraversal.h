#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "WatershedSegmenter.h"



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
		waitKey(); // Wait for a keystroke in the window
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

	//基于标记点的分水岭算法应用
	/*
	算法步骤：
		封装分水岭算法类
		获取标记图像
			获取前景图像，并用255标记
			获取背景图像，并用128标记，未知像素使用0标记
			合成标记图像
		将原图和标记图像输入分水岭算法
		显示结果
	*/
	cv::Mat getMarker(void) {

		// 目的是获取前景物体的像素，并用255标价，这里的阈值分割初步划分出前景与背景，然后使用
		// 形态学 闭运算 连接二值图像中前景的各个部分，平滑边缘；
		cv::Mat binary;
		cv::cvtColor(img, binary, COLOR_BGR2GRAY);
		int thred = 30;
		cv::threshold(binary, binary, thred, 255, THRESH_BINARY_INV);

		// COLSE operation
		cv::Mat elem5(5, 5, CV_8U, cv::Scalar(1)); //5x5全1结构元
		cv::Mat fg;
		cv::morphologyEx(binary, fg, cv::MORPH_CLOSE, elem5, Point(-1, -1), 1);
		
		
		//在二值化图像的基础上， 通过对白色前景做深度膨胀运算火的一个超过前景实际大小的物体，紧接着用反向阈值将膨胀后
		//的图像中的黑色部分转换成灰度128， 即完成了背景元素的标记
		//！！！在0-255范围内，任意不为0或者255的值都可以作为背景的标记，多个目标可以有多个标记
		//来帮助分水岭算法正确分割图像
		cv::Mat bg;
		cv::dilate(binary, bg, cv::Mat(), cv::Point(-1, -1), 4); //3*3结构元膨胀4次
		
		cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

		//合成 将前景、背景以及未知区域合成一个标记图像，标记为255的是前景图像，标记为128的是背景，标记为0的是未知区域
		cv::Mat marker = fg + bg;

		//imshow("前景", fg);
		//imshow("背景", bg);
		//imshow("合成", marker);

		return marker;
	}

	cv::Mat segmentWaterShed(void) {
		cv::Mat marker = getMarker();

		WatershedSegmenter seg;
		seg.setMarkers(marker); //设置算法的标记图像
		seg.process(img); //带有标记图像的分割
		

		//// 返回修改后的标记图
		//marker = seg.getSegmentation();
		//imshow("标记图", marker);
		//waitKey();

		//以分水岭图像返回
		marker = seg.getWatersheds();
		
		return marker;
	}

	void setMaskedImage(void) {
		cv::Mat marker = getMarker();
		WatershedSegmenter seg;
		seg.setMarkers(marker); //设置算法的标记图像
		seg.process(img); //带有标记图像的分割

		cv::Mat mask = seg.getSegmentation();
		cv::threshold(mask, mask, 250, 1, THRESH_BINARY); //前景为1， 其余为0
		cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
		
		mask = img.mul(mask);

		imshow("分割后图像", mask);
		waitKey();


		//反转颜色
		int rows = mask.rows;
		int cols = mask.cols * mask.channels();
		for (int i = 0; i < rows; i++) {
			uchar* rowData = mask.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				if (rowData[j] == 0) rowData[j] = 255;
			}
		}
		imshow("分割后图像(反转颜色)", mask);
		waitKey();

	}

private:
	bool isOpened;
	string filename;
	cv::Mat img;
};