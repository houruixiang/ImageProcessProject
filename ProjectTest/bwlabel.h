#pragma once
#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <set>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
using namespace std;


class BWLable {
public:
	BWLable(void):isOpened(false){
		filename = cv::String("../liantong.jpg");
	}
	BWLable(cv::String& _filename) : isOpened(false), filename(_filename) {
	}
	BWLable(std::string& _filename) : isOpened(false) {
		filename = cv::String(_filename);
	}
	virtual ~BWLable() {
	}
	bool open(void) {
		Mat _img = cv::imread(filename, IMREAD_GRAYSCALE);
		if (_img.empty()) {
			std::cout << "Could not open or find the image" << std::endl;
			isOpened = false;
			return false;
		}
		int threshold_value = 128;
		int max_BINARY_value = 255;
		int threshold_type = 0; // 0: Binary 1: Binary Inverted  2: Truncate  3: To Zero 4: To Zero Inverted
		threshold(_img, img, threshold_value, max_BINARY_value, threshold_type);
		
		isOpened = true;
		return true;
	}

	bool create3Rows(void) {

		static uchar vec[3][14] = {
			{0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff},
			{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00},
			{0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00}
		};
		img = cv::Mat(3, 14, CV_8UC1, vec);
 
		isOpened = true;
		return true;
	}


	bool invertedSave(void) {
		if (!isOpened)  open();
		Mat _img = img;
		threshold(img, _img, 128, 255, 1);
		imwrite(filename, _img);
	}
	void display_img(void) {
		if (!isOpened)  open();
		namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("Display window", img);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
	}

	void display_labelingColorIimg(void) {
		if (colorLabelImg.empty())  std::cout << "Could not open or display the labeled image" << std::endl;
		namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("LabelColoredColor Image", colorLabelImg);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
	}

	cv::Scalar getRandomColor() {
		uchar r = 255 * (rand() / (1.0 + RAND_MAX));
		uchar g = 255 * (rand() / (1.0 + RAND_MAX));
		uchar b = 255 * (rand() / (1.0 + RAND_MAX));
		return cv::Scalar(b, g, r);
	}

	// ��ɫ���
	int labelingColorImg(const cv::Mat& labelImg) {
		int num = 0;
		if (labelImg.empty() || labelImg.type() != CV_32SC1) {
			return num;
		}
		std::map<int, cv::Scalar> colors;
		int rows = labelImg.rows;
		int cols = labelImg.cols;

		colorLabelImg.release();
		colorLabelImg.create(rows, cols, CV_8UC3);
		colorLabelImg.setTo(0);

		for (int i = 0; i < rows; i++) {
			const int* data_src = (int*)labelImg.ptr<int>(i);
			uchar* data_dst = colorLabelImg.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				int pixelValue = data_src[j];
				if (pixelValue >= 1) {
					if (!colors.count(pixelValue)) {
						colors[pixelValue] = getRandomColor();
						num++;
					}
					cv::Scalar color = colors[pixelValue];
					*data_dst++ = color[0];
					*data_dst++ = color[1];
					*data_dst++ = color[2];
				}
				else {
					data_dst++;
					data_dst++;
					data_dst++;
				}
			}
		}
		return num;
	}




	//  1������ɨ��ͼ�����ǰ�ÿһ���������İ�ɫ�������һ�����г�Ϊһ����(run)��
	//     �������������start�������յ�end�Լ������ڵ��кš�
	//	2�����ڳ��˵�һ���������������ţ��������ǰһ���е������Ŷ�û���غ�����
	//     �����һ���µı�ţ������������һ����һ�������غ���������һ�е��Ǹ��ŵı�Ÿ�������
	//     ���������һ�е�2�����ϵ������ص����������ǰ�Ÿ�һ�������ŵ���С��ţ�
	//     ������һ�е��⼸���ŵı��д��ȼ۶ԣ�˵����������һ�ࡣ
	//	3�����ȼ۶�ת��Ϊ�ȼ����У�ÿһ��������Ҫ��һ��ͬ�ı�ţ���Ϊ���Ƕ��ǵȼ۵ġ���1��ʼ����ÿ���ȼ�����һ����š�
	//	4��������ʼ�ŵı�ǣ����ҵȼ����У����������µı�ǡ�
	//	5����ÿ���ŵı��������ͼ���С�
	//	6��������
	int findConnected_TwoPass(cv::Mat& labelImg) {
		int NumberOfRuns = 0;
		vector<int> stRun;
		vector<int> enRun;
		vector<int> rowRun;

		//1��
		fillRunVectors(NumberOfRuns, stRun, enRun, rowRun);

		//2��
		vector<int> runLabels;
		vector<pair<int, int>> equivalences;
		int offset = 0;
		firstPass(stRun, enRun, rowRun, NumberOfRuns,
			runLabels, equivalences, offset);

		//3��
		replaceSameLabel(runLabels, equivalences);

		// ���label
		labelImg.release();
		img.convertTo(labelImg, CV_32SC1);
		labelImg.setTo(0);
		for (int i = 0; i < NumberOfRuns; i++) {
			int y = rowRun[i];
			for (int x = stRun[i]; x <= enRun[i]; x++) {
				labelImg.at<int>(y, x) = runLabels[i];
			}
		}
		return NumberOfRuns;
	}

 
	////Seed-Filling�㷨
	//��1��ɨ��ͼ��ֱ����ǰ���ص�B(x, y) == 1��
	//	   a����B(x, y)��Ϊ���ӣ�����λ�ã�����������һ��label��Ȼ�󽫸��������ڵ�����ǰ�����ض�ѹ��ջ�У�
	//	   b������ջ�����أ���������ͬ��label��Ȼ���ٽ����ջ���������ڵ�����ǰ�����ض�ѹ��ջ�У�
	//	   c���ظ�b���裬ֱ��ջΪ�գ�
	//	      ��ʱ�����ҵ���ͼ��B�е�һ����ͨ���򣬸������ڵ�����ֵ�����Ϊlabel��
	// (2���ظ��ڣ�1������ֱ��ɨ�������
	//	   ɨ������󣬾Ϳ��Եõ�ͼ��B�����е���ͨ����
	//	   ���ӣ�https ://www.jianshu.com/p/faba96cb624a

	void findConnected_SeedFilling(cv::Mat& labelImg, int offset) {
		
		int rows = img.rows;
		int cols  = img.cols;
		int labels = 0;

		labelImg.release();
		img.convertTo(labelImg, CV_32SC1);
		Mat mask(rows, cols, CV_8UC1);
		mask.setTo(0);	
 		for (int i = 0; i < rows; i++) {
			int *data = labelImg.ptr<int>(i);
			uchar *maskPtr = mask.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				if (data[j] == 0xff && maskPtr[j] != 1) {
					maskPtr[j] = 1;
					std::stack<std::pair<int, int>> neighborPixels;
					neighborPixels.push(std::make_pair(i, j));
					++labels;
					while (!neighborPixels.empty()) {
						std::pair<int, int>curPixel = neighborPixels.top();
						neighborPixels.pop();
						int curY = curPixel.first;
						int curX = curPixel.second;
						labelImg.at<int>(curY, curX) = labels;

						//push the 4-neighbors
						if (curX - 1 >= 0) {
							if (labelImg.at<int>(curY, curX - 1) == 0xff && mask.at<uchar>(curY, curX - 1) != 1) {
								neighborPixels.push(std::make_pair(curY, curX-1));
								mask.at<uchar>(curY, curX - 1) = 1;
							}
						}
						if (curX + 1 < cols) {
							if (labelImg.at<int>(curY, curX + 1) == 0xff && mask.at<uchar>(curY, curX + 1) != 1) {
								neighborPixels.push(std::make_pair(curY, curX + 1));
								mask.at<uchar>(curY, curX + 1) = 1;
							}
						}
						if (curY - 1 >= 0) {
							if (labelImg.at<int>(curY-1, curX) == 0xff && mask.at<uchar>(curY - 1, curX) != 1) {
								neighborPixels.push(std::make_pair(curY - 1, curX));
								mask.at<uchar>(curY - 1, curX) = 1;
							}
						}
						if (curY + 1 < rows) {
							if (labelImg.at<int>(curY + 1, curX) == 0xff && mask.at<uchar>(curY + 1, curX) != 1) {
								neighborPixels.push(std::make_pair(curY + 1, curX));
								mask.at<uchar>(curY+1, curX) = 1;
							}
						}
					}
				}
			}
		}
	}
 
private:
	bool isOpened;
	cv::String filename;
	cv::Mat img;
	cv::Mat colorLabelImg;

	void fillRunVectors(int& NumberOfRuns, 
		vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun);
	void firstPass(vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun, int NumberOfRuns,
		vector<int>& runLabels, vector<pair<int, int>>& equivalences, int offset);
	void replaceSameLabel(vector<int>& runLabels,
		vector<pair<int, int>>& equivalence);
};


/* 
��һ������������ŵĲ������¼
	NumberOfRuns�� �ű��
	stRun: ��Run��ʼλ��
	enRun: ��Run�յ�λ��
	rowRun:��Run������
*/
void BWLable::fillRunVectors(int& NumberOfRuns, 
	vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun) {

	int rows = img.rows, cols = img.cols;
	for (int i = 0; i < rows; i++) {
		const uchar* rowData = img.ptr<uchar>(i);
		if (rowData[0] == 0xff) {
			NumberOfRuns++;
			stRun.push_back(0); rowRun.push_back(i);
		}
		for (int j = 1; j < cols; j++) {
			if (rowData[j - 1] == 0xff && rowData[j] == 0) {
				enRun.push_back(j-1);
			}
			else if (rowData[j - 1] == 0 && rowData[j] == 0xff) {
				NumberOfRuns++;
				stRun.push_back(j); rowRun.push_back(i);
			}
		}
		if (rowData[cols - 1] == 0xff) {
			enRun.push_back(cols - 1);
		}
	}
}


//2��firstPass��������ŵı����ȼ۶��б�����ɡ�
/*
	vector<int>& stRun, 
	vector<int>& enRun, 
	vector<int>& rowRun, 
	int NumberOfRuns,
	vector<int>& runLabels, 
	vector<pair<int, int>>& equivalences, 
	int offset 0:4-connected 1:8-connected
*/
void  BWLable::firstPass(vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun, int NumberOfRuns,
	vector<int>& runLabels, vector<pair<int, int>>& equivalences, int offset)
{
	runLabels.assign(NumberOfRuns, 0);
	int idxLabel = 1;
	int curRowIdx = 0;
	int firstRunOnCur = 0;
	int firstRunOnPre = 0;
	int lastRunOnPre = -1;
	for (int i = 0; i < NumberOfRuns; i++)
	{
		// ����Ǹ��еĵ�һ��run���������һ�е�һ��run�����һ��run�����
		if (rowRun[i] != curRowIdx)
		{
			curRowIdx = rowRun[i]; // �����е����
			firstRunOnPre = firstRunOnCur; // ��һ�еĵ�һ��run
			lastRunOnPre = i - 1;          // ��һ�����һ��run   
			firstRunOnCur = i;             // ��ǰ�е�һ��run
		}
		// ������һ�е�����run���ж��Ƿ��ڵ�ǰrun���غϵ�����
		for (int j = firstRunOnPre; j <= lastRunOnPre; j++)
		{
			// �����غ��Ҵ������ڵ�����
			if (stRun[i] <= enRun[j] + offset && enRun[i] >= stRun[j] - offset && rowRun[i] == rowRun[j] + 1)
			{
				if (runLabels[i] == 0) // û�б���Ź�
					runLabels[i] = runLabels[j];
				else if (runLabels[i] != runLabels[j])// �Ѿ������             
					equivalences.push_back(make_pair(runLabels[i], runLabels[j])); // ����ȼ۶�
			}
		}
		if (runLabels[i] == 0) // û����ǰһ�е��κ�run�غ�
		{
			runLabels[i] = idxLabel++;
		}

	}
}

//3) ÿ���ȼ۱���һ��vector<int>�����棬�ȼ۶��б�����map<pair<int,int>>��
void BWLable::replaceSameLabel(vector<int>& runLabels,
	vector<pair<int, int>>& equivalence)
{
	int maxLabel = *max_element(runLabels.begin(), runLabels.end());
	vector<vector<bool>> eqTab(maxLabel, vector<bool>(maxLabel, false));
	vector<pair<int, int>>::iterator vecPairIt = equivalence.begin();
	while (vecPairIt != equivalence.end()) //�����ڽӾ���
	{
		eqTab[vecPairIt->first - 1][vecPairIt->second - 1] = true;
		eqTab[vecPairIt->second - 1][vecPairIt->first - 1] = true;
		vecPairIt++;
	}
	vector<int> labelFlag(maxLabel, 0);
	vector<vector<int>> equaList;
	vector<int> tempList;
	//cout << maxLabel << endl;
	for (int i = 1; i <= maxLabel; i++)
	{
		if (labelFlag[i - 1]) continue;
		labelFlag[i - 1] = equaList.size() + 1;
		tempList.push_back(i);
		for (vector<int>::size_type j = 0; j < tempList.size(); j++)
		{
			for (vector<bool>::size_type k = 0; k != eqTab[tempList[j] - 1].size(); k++)
			{
				if (eqTab[tempList[j] - 1][k] && !labelFlag[k])
				{
					tempList.push_back(k + 1);
					labelFlag[k] = equaList.size() + 1;
				}
			}
		}
		equaList.push_back(tempList);
		tempList.clear();
	}
	//cout << equaList.size() << endl;
	for (vector<int>::size_type i = 0; i != runLabels.size(); i++)
	{
		runLabels[i] = labelFlag[runLabels[i] - 1];
	}
}



