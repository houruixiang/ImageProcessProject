#include <iostream>
#include <vector>
#include <string>
#include "ImageTraversal.h"

// OpenCV成长之路(2)：图像的遍历

void display_img(cv::Mat _img) {
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", _img);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
}


int main(void) {

	cv::String fpath = "./../../img/lena.jpg";
	ImageTraversal image = ImageTraversal(fpath);
	image.open();
	int64 start;
	double duration;


	cv::Mat _img;
	
	
	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_twoRepeatTravel_at(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_twoRepeatTravel_at(ms): " << duration << endl;

	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_twoRepeatTravel_ptr(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_twoRepeatTravel_ptr(ms): " << duration << endl;
	
	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_oneRepeatTravel_iterator(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_oneRepeatTravel_iterator(ms): " << duration << endl;




	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_twoRepeatTravel_at_bit(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_twoRepeatTravel_at_bit(ms): " << duration << endl;

	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_twoRepeatTravel_ptr_bit(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_twoRepeatTravel_ptr_bit(ms): " << duration << endl;

	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_mask_bit(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_mask_bit(ms): " << duration << endl;
	


	_img = image.copy();
	start = cv::getTickCount();
	image.colorReduce_oneRepeatTravel_iterator_bit(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of colorReduce_oneRepeatTravel_iterator_bit(ms): " << duration << endl;
	
	///image.display_img();

	_img = image.copy();
	start = cv::getTickCount();
	image.sharpen(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of sharpen(ms): " << duration << endl;


	_img = image.copy();
	start = cv::getTickCount();
	image.sharpen2D(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of sharpen2D(ms): " << duration << endl;
	
	
	return 0;
}
