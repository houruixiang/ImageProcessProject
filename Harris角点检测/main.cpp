#include <iostream>
#include <vector>
#include <string>
#include "ImageTraversal.h"


int main(void) {

	//cv::String fpath = "./../../img/lena.jpg";
	//ImageTraversal image = ImageTraversal(fpath);
	//image.open();
	//int64 start;
	//double duration;

	//cv::Mat _img = image.copy();
	//start = cv::getTickCount();
	//image.sharpen(_img);
	//duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	//cout << "The time of open(ms): " << duration << endl;
	
	//imshow("Display the processed imgage window", _img); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window

	cv::Mat image = imread("./../../img/lena.jpg", IMREAD_GRAYSCALE);
	HarrisDetector_all(image);
	imshow("½Çµã¼ì²â", image);
	waitKey(0);


	return 0;
}
