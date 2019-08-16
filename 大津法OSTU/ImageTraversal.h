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
	��򷨣�
		��򷨾��������䷽�ͨ��ͳ��ͼ��ֱ��ͼ��Ϣ���Զ�ȷ����ֵ���Ӷ�����ǰ���뱳��
	�㷨���̣�
		1. ����ͼ�����أ�ͳ��ÿ������ֵ���ֵĴ�������255��bin��ͳ��ÿ��bin���صĸ�����
		2. ����0��255����ֵ����i����ֵΪ���ֵ�ǰ�������ֵ���ڣ�0-i)��Χ�ڼ���ǰ������
		ƽ���Ҷ�u0, ǰ���������ص���ռ����w0,�ڣ�i~255�����㱳�����ر�������ƽ���Ҷ�u1��
		�����������ص���ռ����w1;
		3. ���㵱ǰ��䷽�� varValue = w0*w1*(u1-u0)*(u1-u0);
		4. ѡ�������䷽��varValueʱ��i������ֵ��Ϊ����ǰ���뱳���������ֵ��
	*/
	int Ostu_threshold() {
		int threshold = 0;
		double maxVarValue = 0; //
		double w0 = 0, w1 = 0;  //������ǰ�����ص���ռ����
		double u0 = 0, u1 = 0;  //������ǰ�����ص�ƽ���Ҷ�
		double histgram[256] = { 0 }; 
		double numPixels = static_cast<double>(img.rows) * img.cols;

		//ͳ��256bin�и��Ե����ظ���
		for (int i = 0; i < img.rows; i++) {
			const uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				histgram[rowData[j]]++;
			}
		}

		//ǰ������ͳ��
		for (int i = 0; i < 255; i++) {
			w0 = 0, w1 = 0; 
			u0 = 0, u1 = 0;
			for (int j = 0; j <= i; j++) {
				w0 += histgram[j]; //��iΪ��ֵ��ͳ��ǰ�����صĸ���
				u0 += j * histgram[j]; //��iΪ��ֵ������ǰ�����ص��ܻҶȺ�
			}
			u0 /= w0; //ǰ������ƽ���Ҷ�
			w0 /= numPixels; //ǰ��������ռ����
			

			for (int j = i + 1; j <= 255; j++) {
				w1 += histgram[j]; //�������ظ���
				u1 += j * histgram[j]; //�������ػҶ��ܺ�
			}
			u1 /= w1; //��������ƽ���Ҷ�
			w1 /= numPixels; // ƽ��������ռ��ֵ

			double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //��ǰ��䷽��
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
		double w0 = 0, w1 = 0;  //������ǰ�����ص���ռ����
		double u0 = 0, u1 = 0;  //������ǰ�����ص�ƽ���Ҷ�
		double histgram[256] = { 0 };
		double numPixels = static_cast<double>(img.rows) * img.cols;

		//
		double allPixelsGray = 0;
		//ͳ��256bin�и��Ե����ظ���
		for (int i = 0; i < img.rows; i++) {
			const uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				histgram[rowData[j]]++;
				allPixelsGray += rowData[j];
			}
		}

		//ǰ������ͳ��
		for (int i = 0; i < 255; i++) {
			w0 = 0, w1 = 0;
			u0 = 0, u1 = 0;
			for (int j = 0; j <= i; j++) {
				w0 += histgram[j]; //��iΪ��ֵ��ͳ��ǰ�����صĸ���
				u0 += j * histgram[j]; //��iΪ��ֵ������ǰ�����ص��ܻҶȺ�
			}
			
			u1 = (allPixelsGray - u0) / (numPixels - w0); //��������ƽ���Ҷ�
			u0 /= w0; //ǰ������ƽ���Ҷ�
			w0 /= numPixels; //ǰ��������ռ����
			w1 = 1 - w0; // ƽ��������ռ����
			

			double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //��ǰ��䷽��
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
		double w0 = 0, w1 = 0;  //������ǰ�����ص���ռ����
		double u0 = 0, u1 = 0;  //������ǰ�����ص�ƽ���Ҷ�
		double histgram[256] = { 0 };
		double numPixels = static_cast<double>(img.rows) * img.cols;

		//
		double allPixelsGray = 0;
		//ͳ��256bin�и��Ե����ظ���
		for (int i = 0; i < img.rows; i++) {
			const uchar* rowData = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++) {
				histgram[rowData[j]]++;
				allPixelsGray += rowData[j];
			}
		}

		//ǰ������ͳ��
		double w0_accum = 0;
		double u0_accum = 0;
		for (int i = 0; i < 255; i++) {
			w0_accum += histgram[i]; //��iΪ��ֵ��ͳ��ǰ�����صĸ���
			u0_accum += i * histgram[i]; //��iΪ��ֵ������ǰ�����ص��ܻҶȺ�
 
			w0 = w0_accum; 
			u0 = u0_accum;
			u1 = (allPixelsGray - u0) / (numPixels - w0); //��������ƽ���Ҷ�
			u0 /= w0; //ǰ������ƽ���Ҷ�
			w0 /= numPixels; //ǰ��������ռ����
			w1 = 1 - w0; // ƽ��������ռ����


			double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //��ǰ��䷽��
			if (maxVarValue < variance) {
				maxVarValue = variance;
				threshold = i;
			}
		}
		return threshold;
	}



	/*
	��ֵ��
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