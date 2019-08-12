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
	int size() { return img.rows * img.cols; }
	cv::Mat copy(void) { return img.clone(); }
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


	//�Ҷ�ͼ
	//�Ҷ�ͼ����RGBͼ���ת�䡣
	//RGB[A]ת��Ϊ�Ҷȣ�Y = 0.299 * R + 0.587 * G + 0.114 * B;
	//��cvtColor������ȱ�ݾ����ǵ�ԭʼͼ��ĶԱȶȶ�ʧ���������ʽ���п��Կ�����
	void toGray() {
		if (img.type() == CV_8UC3) {
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
	}
	//���������
	void addSaltNoise(int num) {
		//�����������
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

	//��ӽ�����
	void addPepperNoise(int num) {
		//�����������
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
	��ֵ�˲���˼����ǱȽ�һ�������ڵ�����ֵ�Ĵ�С��ȡ������ֵ��Ϊ�����������������µ�ֵ��
	�����һ�������ڵ��������ش�С�����������������ڹ����������㣬���罷��������������������С�ĻҶ�ֵ��
	���ֵ�Ч����С�ڵ㣻�����������ϴ�ĻҶ�ֵ�����ֵ�Ч����С�׵㣩����ô��С�����������������У�
	��Щ����������һ����ֲ������ߣ�Ҫô��С��Ҫô�ܴ󣩣�������ȡ������ֵ����Ժܺõر���������Ϣ�����˳����������Ӱ�졣 
    ��ֵ�˲������˲����ڴ�СӰ��ϴ��������������ͱ���ͼ��ϸ�ڣ����߻���ڳ�ͻ��������ڽ�С��
	���ܽϺõر���ͼ���е�һЩϸ����Ϣ�����������Ĺ���Ч���ͻ���ۿۣ���֮��������ڳߴ�ϴ�����нϺõ���������Ч����
	��Ҳ���ͼ�����һ����ģ��Ч�����Ӷ���ʧһ����ϸ����Ϣ�����⣬������˲������ڵ�������ĸ��������������������صĸ�����
	����ֵ�˲��Ͳ��ܺܺõĹ��˵�������

	��ɫͼ��һ�㲻����rgb�ռ�������ֵ�˲���������ﴦ����ǻҶ�ͼ��
	*/
	uchar medianFilter(cv::Mat& _img, int row, int col, int kernelSize) {
		std::vector<uchar> pixels;
		int radius = kernelSize / 2;
		for (int y = -radius; y <= radius; y++) {
			uchar* rowData = _img.ptr<uchar>(row+y);
			for (int x = -radius; x <= radius; x++) {
				pixels.push_back(rowData[col+x]);
			}
		}
		
		sort(pixels.begin(), pixels.end());
		return pixels[pixels.size() / 2];
	}
	// ����㷨Ч�ʼ������
	void medianFilter(cv::Mat& _img, int kernelSize) {
		int raduis = kernelSize / 2;
		// ��չͼ��ı߽�
		cv::copyMakeBorder(img, _img, raduis, raduis, raduis, raduis, cv::BORDER_REFLECT);
		int rows = _img.rows;
		int cols = _img.cols;
		for (int y = raduis; y < rows - raduis; y++) {
			for (int x = raduis; x < cols - raduis; x++) {
				img.at<uchar>(y-raduis, x-raduis) = medianFilter(_img, y, x, kernelSize);
			}
		}

		////�߽紦��
		//for (int y = 0; y < raduis; y++) img.row(y).setTo(0);
		//for (int y = rows - raduis; y < rows; y++) img.row(y).setTo(cv::Scalar(0));
		//for (int x = 0; x < raduis; x++) img.col(x).setTo(0);
		//for (int x = cols-raduis; x < cols; x++) img.col(x).setTo(cv::Scalar(0));
	}
	
	/*
	�������ܶȲ��Ǻܴ������£����ݾ��飬�����ĳ��ֵĸ���С��0.2����ʹ����ֵ�˲���Ч������
	���ǵ��������ֵĸ��ʱȽϸ�ʱ��ԭ������ֵ�˲��㷨�Ͳ��Ǻ���Ч�ˡ�ֻ�������˲������ڳߴ磬���ܻ�ʹͼ����ģ���� 
	ʹ������Ӧ��ֵ�˲�����Ŀ�ľ��ǣ�����Ԥ��õ���������̬�ظı���ֵ�˲����Ĵ��ڳߴ磬��ͬʱ���ȥ�������úͱ���ϸ�ڵ�Ч���� 
 	*/
	uchar adaptiveMedianFilter(cv::Mat& _img, int row, int col, int kernelSize) {
		std::vector<uchar> pixels;
		int raduis = kernelSize / 2;
		for (int y = -raduis; y <= raduis; y++) {
			uchar* rowData = _img.ptr<uchar>(row + y);
			for (int x = -raduis; x <= raduis; x++) {
				pixels.push_back(rowData[col + x]);
			}
		}
		sort(pixels.begin(), pixels.end());
		auto min = pixels.front();
		auto max = pixels.back();
		auto mid = pixels[pixels.size() / 2];
		auto zxy = _img.at<uchar>(row, col);

		if (min < mid && mid < max) { //mid �ǽ���
			if (min < zxy && zxy < max) //hold
				return zxy;
			else
				return mid;
		}
		else {
			kernelSize += 2;
			if (kernelSize <= maxSize)
				return adaptiveMedianFilter(_img, row, col, kernelSize);
			else
				return mid;
		}
	}
	void adaptiveMedianFilter(cv::Mat& _img) {
		int raduis = maxSize / 2;
		// ��չͼ��ı߽�
		cv::copyMakeBorder(img, _img, raduis, raduis, raduis, raduis, cv::BORDER_REFLECT);
		int rows = img.rows;
		int cols = img.cols;
		for (int y = raduis; y < rows - raduis; y++) {
			for (int x = raduis; x < cols - raduis; x++) {
				img.at<uchar>(y- raduis, x- raduis) = adaptiveMedianFilter(_img, y, x, minSize);
			}
		}

		////�߽紦��
		//for (int y = 0; y < raduis; y++) img.row(y).setTo(0);
		//for (int y = rows - raduis; y < rows; y++) img.row(y).setTo(cv::Scalar(0));
		//for (int x = 0; x < raduis; x++) img.col(x).setTo(0);
		//for (int x = cols - raduis; x < cols; x++) img.col(x).setTo(cv::Scalar(0));
	}

private:
	int minSize, maxSize;
	bool isOpened;
	string filename;
	cv::Mat img;
	cv::Mat img_backup;
};