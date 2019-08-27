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
	//32λ����Ľǵ�ǿ��ͼ��
	cv::Mat cornerStrength;
	//32λ�������ֵ��ͼ��
	cv::Mat cornerTh;
	//�ֲ����ֵͼ��
	cv::Mat localMax;
	//ƽ��ƫ����������߶�
	int neighborhood;
	//�ݶȼ���Ŀھ�
	int aperture;
	//Harris����
	double k;
	//��ֵ��������ǿ��
	double maxStrength;
	//����õ�����ֵ
	double threshold;
	//�����ֵ���Ƶ�����ߴ�
	int nonMaxSize;
	//�����ֵ���Ƶ��ں�
	int nonMaxKernel;
public:
	HarrisDetector() : neighborhood(3), aperture(3),
		k(0.01), maxStrength(0.0),
		threshold(0.01), nonMaxSize(3) {

	}

	// �ǵ���
	void detect(const cv::Mat& image) {
		//����Harris
		cv::cornerHarris(image, cornerStrength,
			neighborhood, aperture, k);
		//�����ڲ���ֵ���ֵ
		cv::minMaxLoc(cornerStrength, 0, &maxStrength);

		//�Ǽ���ֵ���ƣ��������ų������ڵ�Harris�ǵ㣬��Harris�ǵ㲻��Ҫ��
		//����ָ����ֵ�ĵ÷֣��������Ǿֲ���Χ�ڵ����ֵ
		//����������������ڰ�ÿ�����ض��滻�����ֵ��ֻ�оֲ����ֵ�ǲ���ģ�
		//localMax����ֻ���ھֲ����ֵλ���ϲ����棬
		cv::Mat dilated; //����ͼ��
		cv::dilate(cornerStrength, dilated, cv::Mat());
		cv::compare(cornerStrength, dilated, localMax, cv::CMP_EQ);

	}

	//��Harrisֵ�õ��ǵ�ֲ�ͼ
	cv::Mat getCornerMap(double qualityLevel) {
		cv::Mat cornerMap;
		threshold = qualityLevel * maxStrength;
		cv::threshold(cornerStrength, cornerTh, threshold, 255, cv::THRESH_BINARY); //��ֵ��

		//ת����8bitͼ��
		cornerTh.convertTo(cornerMap, CV_8U);

		//�Ǽ���ֵ����
		cv::bitwise_and(cornerMap, localMax, cornerMap);

		return cornerMap;
	}

	//������ϵ�е��������ΪͼƬ�����Ͻǣ�X��Ϊͼ����ε���������ˮƽ�ߣ�Y��Ϊͼ�������ߵ�������ֱ�ߡ���������ϵ������ṹ��Mat, Rect, Point�ж������õġ�
	//��Ȼ������ѧ��˵OpenCV����Щ���ݽṹ������ԭ������ͼƬ�����½ǣ���������ʱ��û����������

	//��ʹ��image.at(x1, x2)������ͼ���е��ֵ��ʱ��x1������ͼƬ�ж�Ӧ���x�����꣬����ͼƬ�ж�Ӧ���y���ꡣ
	//�������ʵĽ����ʵ�Ƿ���imageͼ���е�Point(x2, x1)�㣬����image.at(Point(x2, x1))Ч����ͬ��

	//�������ͼ���Ƕ�ͨ���ģ�����˵imageͼ���ͨ����ʱn����ʹ��Mat::at(x, y)ʱ����x�ķ�Χ������0��image��height��
	//��y��ȡֵ��Χ����0��image��width����n����Ϊ���ʱ������n��ͨ��������ÿ��������Ҫռ��n�С����������ͬ��������£�
	//ʹ��Mat::at(point)�����ʵĻ�������ʱ����Բ��ÿ���ͨ���ĸ�������Ϊ��Ҫ��ֵ����ȡMat::at(point)��ֵʱ��
	//������һ�����֣�����һ����Ӧ��nά������
	//�ýǵ�ֲ�ͼ�õ�������
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

	//���������
	void getCorners(std::vector<cv::Point>& points, double qualityLevel) {
		//��ýǵ�ֲ�ͼ
		cv::Mat cornerMap = getCornerMap(qualityLevel);
		//��ýǵ�
		getCorners(points, cornerMap);
	}

	//��������λ�û�Բ��
	void drawOnImage(cv::Mat& image, const std::vector<cv::Point>& points,
		cv::Scalar color = cv::Scalar(255, 255, 255),
		int raduis = 3, int thickness = 1) {
		int lens = points.size();
		//�����нǵ�
		for(int i=0; i<lens; i++)
			cv::circle(image, points[i], raduis, color, thickness);
	}
	 
};


void HarrisDetector_all(cv::Mat& image) {
	HarrisDetector harris;
	//����harrisֵ
	harris.detect(image);
	//���Harris�ǵ�
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


	// ---------------------------����������˹���ӵ���ǿ--------------------------------------
	/*
	 ԭʼͼ���ȥ������˹ͼ��           ������˹ͼ��
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