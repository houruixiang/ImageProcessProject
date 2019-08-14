#include <iostream>
#include <vector>
#include <string>
#include "ImageTraversal.h"


int main(void) {

	cv::String fpath = "./../../img/color.jpg";
	ImageTraversal image = ImageTraversal(fpath);
	image.open();
	int64 start;
	double duration;

	cv::Mat _img = image.copy();
	start = cv::getTickCount();
	image.sharpen(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of open(ms): " << duration << endl;
	
	double minHue = 160; 
	double maxHue = 50;
	double minSat = 80;
	double maxSat = 150;
	cv::Mat mask;
	image.detectHScolor(minHue, maxHue, minSat, maxSat, mask);
	imshow("mask", mask);
	cv::Mat detected(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	_img.copyTo(detected, mask);
	
	imshow("Display the detectHScolor imgage window", detected); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window


	return 0;
}
