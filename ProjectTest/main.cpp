#include "bwDFS.h"
#include "bwlabel.h"


#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
 


int main(void)
{
	int64 num, start;
	double duration;
	cv::String fpath = "../test.jpg";
	BWLable bw = BWLable(fpath);
	bw.open();
	//bw.create3Rows();
	//bw.invertedSave();
	cv::Mat labelImg;

	//test the performance of findConnected_SeedFilling
	start = cv::getTickCount();
	bw.findConnected_SeedFilling(labelImg, 0);
	duration = 1000.0*(cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of findConnected_SeedFilling(ms): " << duration << endl;
	num = bw.labelingColorImg(labelImg);
	cout << "The number of findConnected_SeedFilling: " << num << endl;
	bw.display_img();
	bw.display_labelingColorIimg();
	
	
	//test the performance of findConnected_SeedFilling
	start = cv::getTickCount();
	bw.findConnected_TwoPass(labelImg);
	duration = 1000.0*(cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of findConnected_TwoPass(ms): " << duration << endl;
	num = bw.labelingColorImg(labelImg);
	cout << "The number of findConnected_TwoPass: " << num << endl;
	bw.display_labelingColorIimg();	
	
	
	system("pause");
	return 0;
}


 