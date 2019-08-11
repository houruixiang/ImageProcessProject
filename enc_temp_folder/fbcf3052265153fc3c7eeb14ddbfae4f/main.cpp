#include <iostream>
#include <vector>
#include <string>
#include "ImageTraversal.h"


int main(void) {
	int64 start;
	double duration;

	cv::String fpath = "./../../img/lena.jpg";
	ImageTraversal image = ImageTraversal(fpath);
	image.open();


	image.toGray();
	cv::Mat _img = image.copy();

	double noiseRate = 0.2;
	int noiseNum = image.size() * noiseRate;
	start = cv::getTickCount();
	image.addPepperNoise(noiseNum);
	image.addSaltNoise(noiseNum);
	
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of open(ms): " << duration << endl;
	//image.display_img();
	
	
	
	
	start = cv::getTickCount();
	image.adaptiveMedianFilter();
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of adaptiveMedianFilter(ms): " << duration << endl;
	image.display_img();

	int kernelSize = 3;
	start = cv::getTickCount();
	image.medianFilter(kernelSize);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of medianFilter(ms): " << duration << endl;
	image.display_img();


	//imshow("Display the imgage window", _img); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window


	return 0;
}
