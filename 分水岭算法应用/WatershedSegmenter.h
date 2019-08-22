#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

/*
输入标记图像可以可以含有任意数值的标签，未知标签的像素值为0， 标记图像的类型
选用32位有符号整数，以便定义超过255个标签，分水岭的对应像素设为特殊值-1；
*/
class WatershedSegmenter {
private:
	cv::Mat markers;

public:
	void setMarkers(const cv::Mat& markerImage) {
		// convert to image of ints
		markerImage.convertTo(markers, CV_32S);
	}

	cv::Mat process(const cv::Mat& image) {
		// Apply watershed
		cv::watershed(image, markers);
		return markers;
	}

	cv::Mat getSegmentation() {
		cv::Mat tmp;
		// all segment with label higher than 255 will be assigned value 255
		markers.convertTo(tmp, CV_8U);
		return tmp;
	}

	//return watershed in the form of an image
	cv::Mat getWatersheds() {
		cv::Mat tmp;
		// 变换前，将每个像素p转换为255p+255,正好将分水岭-1转换为0;
		// 值大于255的像素赋值为255，满足饱和运算
		markers.convertTo(tmp, CV_8U, 255, 255);
 		return tmp;
	}

};