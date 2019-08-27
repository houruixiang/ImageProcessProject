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

	cv::Mat _img = image.copy();
	start = cv::getTickCount();
	image.fast_detect_();
	duration = 1000.0 * (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "The time of open(ms): " << duration << endl;
	
	
	
	
	imshow("Display the processed imgage window", _img); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window


	return 0;
}

 

//���ٶ�˵�������ִ������ڴ����⣬Ҫô�����ڴ�ľ���ͷţ�Ҫô��������ָ��ָ��ͬһ���ڴ棬һ��ָ������ͷŵ��˵ȵȣ�
//�������ڴ����������˺ܾã��������� Ҳû�н��������ҷ����ҰѴ����е�vector
//
//std::vector<cv::KeyPoint> keypoints1, keypoints2;
//cv::Mat descriptors1, descriptors2;
//std::vector<cv::DMatch> matches;//ƥ����
//std::vector<cv::DMatch> goodMatches;
//
//�Ӻ������ó�������Ϊȫ�ֱ��������⾡Ȼ����ˣ���ʵ�Ҷ�û����������ʲôԭ��
//vector������ڴ治�ǲ���Ҫ�ֶ��ͷŵ��𣬻�˵�����ں���������Ϊ�ֲ�������ʱ����Ҳ���Թ��ͷ�����
//����clear����������clear�����ͷ����ڴ棬�����Ұ�������˵����swap���ֶ��ͷţ��������ջ���û�á�
//��������ײ�ģ�������Ϊȫ�ֱ���������ŵ��Խ������������˾ͺá�
//��debug���³����Ѿ����������ˣ�ȫ��ͨ���ˣ��ɽ�����������release���������������ˣ������������£�
//
//
//
//����˵�����������opencv���ʱ����Ӵ��ˣ�����ϸ�����ҵ�additional dependencies����ӣ�
//��release������ȷʵû��d��������ô���ǳ��������ִ����أ�Ҫ���ǳ��������⣬����ôdebug���¿��������أ�
//�Һ����ƣ������˺ܶ෽������������Ե�̬�ȣ��ɴ��release��debug���¶�����˴�d�Ŀ��ļ���
//����������⾡Ȼ����ˡ���������ֻ�ܱ�ʾ�൱���ﰡ��