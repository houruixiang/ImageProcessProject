#include <iostream>
#include <vector>
#include <string>
#include "ImageTraversal.h"
 


int main(void) {

	cv::String fpath = "./../../img/lena.jpg";
	ImageTraversal image = ImageTraversal(fpath);
	image.open();
	int64 start;
	double duration;

	int kernelSize = 5;
	cv::Mat _img;
	start = cv::getTickCount();
	image.meanFilter(_img, kernelSize);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of meanFilter(ms): " << duration << endl;
	image.display_img();

 

	image.recover();
	start = cv::getTickCount();
	image.meanFilter_blur(kernelSize);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of meanFilter_blur(ms): " << duration << endl;
	image.display_img();
	

	image.recover();
	int mu = 0, sigma = 10;
	start = cv::getTickCount();
	image.addGaussNoise(mu, sigma);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of addGaussNoise(ms): " << duration << endl;
	image.display_img();



	image.recover();
	start = cv::getTickCount();
	image.GaussFilter(_img, kernelSize, 1);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of GaussFilter(ms): " << duration << endl;
	image.display_img();


	image.recover();
	start = cv::getTickCount();
	image.GaussFilter_blur(kernelSize, 1.5);
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of GaussFilter_blur(ms): " << duration << endl;
	image.display_img();
	


	imshow("Display the processed imgage window", _img); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window


	return 0;
}
