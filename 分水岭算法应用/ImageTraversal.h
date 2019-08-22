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

	//���ڱ�ǵ�ķ�ˮ���㷨Ӧ��
	/*
	�㷨���裺
		��װ��ˮ���㷨��
		��ȡ���ͼ��
			��ȡǰ��ͼ�񣬲���255���
			��ȡ����ͼ�񣬲���128��ǣ�δ֪����ʹ��0���
			�ϳɱ��ͼ��
		��ԭͼ�ͱ��ͼ�������ˮ���㷨
		��ʾ���
	*/
	cv::Mat getMarker(void) {

		// Ŀ���ǻ�ȡǰ����������أ�����255��ۣ��������ֵ�ָ�������ֳ�ǰ���뱳����Ȼ��ʹ��
		// ��̬ѧ ������ ���Ӷ�ֵͼ����ǰ���ĸ������֣�ƽ����Ե��
		cv::Mat binary;
		cv::cvtColor(img, binary, COLOR_BGR2GRAY);
		int thred = 30;
		cv::threshold(binary, binary, thred, 255, THRESH_BINARY_INV);

		// COLSE operation
		cv::Mat elem5(5, 5, CV_8U, cv::Scalar(1)); //5x5ȫ1�ṹԪ
		cv::Mat fg;
		cv::morphologyEx(binary, fg, cv::MORPH_CLOSE, elem5, Point(-1, -1), 1);
		
		
		//�ڶ�ֵ��ͼ��Ļ����ϣ� ͨ���԰�ɫǰ�����������������һ������ǰ��ʵ�ʴ�С�����壬�������÷�����ֵ�����ͺ�
		//��ͼ���еĺ�ɫ����ת���ɻҶ�128�� ������˱���Ԫ�صı��
		//��������0-255��Χ�ڣ����ⲻΪ0����255��ֵ��������Ϊ�����ı�ǣ����Ŀ������ж�����
		//��������ˮ���㷨��ȷ�ָ�ͼ��
		cv::Mat bg;
		cv::dilate(binary, bg, cv::Mat(), cv::Point(-1, -1), 4); //3*3�ṹԪ����4��
		
		cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

		//�ϳ� ��ǰ���������Լ�δ֪����ϳ�һ�����ͼ�񣬱��Ϊ255����ǰ��ͼ�񣬱��Ϊ128���Ǳ��������Ϊ0����δ֪����
		cv::Mat marker = fg + bg;

		//imshow("ǰ��", fg);
		//imshow("����", bg);
		//imshow("�ϳ�", marker);

		return marker;
	}

	cv::Mat segmentWaterShed(void) {
		cv::Mat marker = getMarker();

		WatershedSegmenter seg;
		seg.setMarkers(marker); //�����㷨�ı��ͼ��
		seg.process(img); //���б��ͼ��ķָ�
		

		//// �����޸ĺ�ı��ͼ
		//marker = seg.getSegmentation();
		//imshow("���ͼ", marker);
		//waitKey();

		//�Է�ˮ��ͼ�񷵻�
		marker = seg.getWatersheds();
		
		return marker;
	}

	void setMaskedImage(void) {
		cv::Mat marker = getMarker();
		WatershedSegmenter seg;
		seg.setMarkers(marker); //�����㷨�ı��ͼ��
		seg.process(img); //���б��ͼ��ķָ�

		cv::Mat mask = seg.getSegmentation();
		cv::threshold(mask, mask, 250, 1, THRESH_BINARY); //ǰ��Ϊ1�� ����Ϊ0
		cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
		
		mask = img.mul(mask);

		imshow("�ָ��ͼ��", mask);
		waitKey();


		//��ת��ɫ
		int rows = mask.rows;
		int cols = mask.cols * mask.channels();
		for (int i = 0; i < rows; i++) {
			uchar* rowData = mask.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				if (rowData[j] == 0) rowData[j] = 255;
			}
		}
		imshow("�ָ��ͼ��(��ת��ɫ)", mask);
		waitKey();

	}

private:
	bool isOpened;
	string filename;
	cv::Mat img;
};