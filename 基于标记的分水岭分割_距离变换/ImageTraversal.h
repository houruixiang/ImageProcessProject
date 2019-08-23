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


	/*
		����RGB��ɫͼ��
		�ҶȻ�����˹�˲���Canny��Ե���
		�������������������б��
		���ڱ�ǵķ�ˮ���㷨
		��ÿ��������������ɫ���
	*/
	cv::Mat autoWaterShed(void){
		cv::Mat imageGray;
		cv::cvtColor(img, imageGray, CV_RGB2GRAY); //�Ҷ�ת��
		
		GaussianBlur(imageGray, imageGray, Size(5, 5), 2); //��˹�˲�
		imshow("Gray Image", imageGray);

	

		//��̬ѧ�ϸ�ʴ���� Ч������
		//cv::dilate(imageGray, imageGray, cv::Mat(), Point(-1, -1), 4);
		//imshow("erode Image", imageGray);  

		////��̬ѧ�Ͽ�����
		//cv::Mat elem5(3, 3, CV_8U, cv::Scalar(1)); //5x5�ĽṹԪ��Ԫ��Ϊ1
		//cv::morphologyEx(imageGray, imageGray, cv::MORPH_OPEN, elem5);
		//imshow("close Image", imageGray);

		////����任 ���ڼ���ͼ����ÿһ���������� �� �Լ���������ľ���
		////�ڶ���mat�������������ÿһ�������Լ���������ľ�����Ϣ��ͼ���ϵĵ�
		////Խ���������˾������ľ���ԽԶ���򵥵���ֵ�Ƚ���ϸ���ַ�������
		//cv::Mat imageThin(imageGray.size(), CV_32FC1);
		//cv::distanceTransform(imageGray, imageThin, CV_DIST_L2, 3);
		//imshow("Thin Image", imageThin);

	
		//float maxDis = 0.0;
		//for (int i = 0; i < imageThin.rows; i++) {
		//	float* data = imageThin.ptr<float>(i);
		//	for (int j = 0; j < imageThin.cols; j++) {
		//		if (data[j] > maxDis) maxDis = data[j]; //��ȡ����任�еļ���ֵ
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


		////��������
  //      vector<vector<Point>> contours;    
		//vector<Vec4i> hierarchy;    
		//cv::findContours(imageGray,contours,hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE,Point());
		//Mat imageContours=Mat::zeros(imageGray.size(),CV_8UC1);  //����     
		//Mat marks(imageGray.size(),CV_32S);   //Opencv��ˮ��ڶ����������  
		//marks=Scalar::all(0);  
		//int index = 0;  
		//int compCount = 0;  
		//for( ; index >= 0; index = hierarchy[index][0], compCount++ )   
		//{  
		//	// ��marks���б�ǣ��Բ�ͬ������������б�ţ��൱������עˮ�㣬�ж������������ж���עˮ��  
		//	drawContours(marks, contours, index, Scalar::all(uchar(compCount + 1)), 1, 8, hierarchy);
		//	drawContours(imageContours,contours,index,Scalar(255),1,8,hierarchy);    
		//}  

		//cv::Mat marksShows;
		//convertScaleAbs(marks, marksShows);
		//imshow("marksShow", marksShows);
		//imshow("����", imageContours);

		return imageGray;
	
	}

private:
	bool isOpened;
	string filename;
	cv::Mat img;
};