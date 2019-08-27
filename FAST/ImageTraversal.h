#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"



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


	//FAST���ٷָ���Ի������ features from accelerated segment test
	void fast_detect_() {
	
		cv::Ptr<cv::FastFeatureDetector> ptrFAST =
			cv::FastFeatureDetector::create(40);
		//���ؼ���
		ptrFAST->detect(img, Keypoints);
		//�����ؼ���
		cv::drawKeypoints(img,
			Keypoints,
			img,
			cv::Scalar(255, 0, 255),
			cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

		imshow("Fast������", img);
		waitKey();
		destroyAllWindows();//�ֶ�����ȫ������
	}

	//������FAST
	void fast_detect_grid(void) {
		cv::Ptr<cv::FastFeatureDetector> ptrFAST =
			cv::FastFeatureDetector::create(30);
		//���ؼ���,������
		int vstep = 32;
		int hstep = 32;
		int vsize = img.rows / vstep;
		int hsize = img.cols / hstep;
		int gridTotal = 20;
		std::vector<cv::KeyPoint>gridPoints;
		for (int i = 0; i < vstep; i++) {
			for (int j = 0; j < hstep; j++) {

				gridPoints.clear();
				cv::Mat imageROI = img(cv::Rect(j * hsize, i * vsize, hsize, vsize));
				
				ptrFAST->detect(imageROI, gridPoints);

				//���ǿ������topK��FAST����
				auto itEnd(gridPoints.end());
				if (gridPoints.size() > gridTotal) {
					std::nth_element(gridPoints.begin(), gridPoints.begin() + gridTotal, gridPoints.end(),
						[](cv::KeyPoint& a, cv::KeyPoint& b) {
							return a.response > b.response;
						});
					itEnd = gridPoints.begin() + gridTotal;
				}
				//����ȫ����������
				for (auto it = gridPoints.begin(); it != itEnd; it++) {
					it->pt += cv::Point2f(j*hsize, i*vsize);
					Keypoints.push_back(*it);
				}
			}
		}


		//�����ؼ���
		cv::drawKeypoints(img,
			Keypoints,
			img,
			cv::Scalar(255, 0, 255),
			cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

		imshow("Fast����������", img);
		waitKey();
		destroyAllWindows();//�ֶ�����ȫ������
	}

private:
	bool isOpened;
	string filename;
	cv::Mat img;	std::vector<cv::KeyPoint>Keypoints;
};