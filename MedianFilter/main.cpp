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

	double noiseRate = 0.5;
	int noiseNum = image.size() * noiseRate;
	start = cv::getTickCount();
	image.addPepperNoise(noiseNum);
	image.addSaltNoise(noiseNum);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of addPepperSaltNoise(ms): " << duration << endl;
	//image.display_img();
	
	cv::Mat _img;

	// 中值滤波
	int kernelSize = 3;
	start = cv::getTickCount();
	image.medianFilter(_img, kernelSize);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of medianFilter(ms): " << duration << endl;
	image.display_img();


	// 自适应中值滤波
	image.recover();
	start = cv::getTickCount();
	image.adaptiveMedianFilter(_img);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of adaptiveMedianFilter(ms): " << duration << endl;
	image.display_img();

	//imshow("Display the imgage window", _img); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window


	return 0;
}
