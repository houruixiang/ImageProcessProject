#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

/*
������ͼ����Կ��Ժ���������ֵ�ı�ǩ��δ֪��ǩ������ֵΪ0�� ���ͼ�������
ѡ��32λ�з����������Ա㶨�峬��255����ǩ����ˮ��Ķ�Ӧ������Ϊ����ֵ-1��
*/
class WatershedSegmenter {
private:
	cv::Mat markers;

public:
	void setMarkers(const cv::Mat& markerImage) {
		// convert to image of ints
		markerImage.convertTo(markers, CV_32S);
	}

	cv::Mat process(const cv::Mat& image) {
		// Apply watershed
		cv::watershed(image, markers);
		return markers;
	}

	cv::Mat getSegmentation() {
		cv::Mat tmp;
		// all segment with label higher than 255 will be assigned value 255
		markers.convertTo(tmp, CV_8U);
		return tmp;
	}

	//return watershed in the form of an image
	cv::Mat getWatersheds() {
		cv::Mat tmp;
		// �任ǰ����ÿ������pת��Ϊ255p+255,���ý���ˮ��-1ת��Ϊ0;
		// ֵ����255�����ظ�ֵΪ255�����㱥������
		markers.convertTo(tmp, CV_8U, 255, 255);
 		return tmp;
	}

};