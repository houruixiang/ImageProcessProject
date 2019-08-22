#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

// �ѵ���
void heapAdjust(vector<int>& nums, int i, int size) {
	if (i > size / 2) return; // ���i��Ҷ�ڵ����ý��е���
	
	int lChild = 2 * i;
	int rChild = 2 * i + 1;
	int max = i;

	if (lChild <= size && nums[max] < nums[lChild]) {
		max = lChild;
	}
	if (rChild <= size && nums[max] < nums[rChild]) {
		max = rChild;
	}
	if (max != i) {
		swap(nums[i], nums[max]);
		heapAdjust(nums, max, size); //�������֮����maxΪ���ڵ���������Ǵ󶥶�
	}
 
}

// ����
void buildHeap(vector<int>& nums, int size) {
	for (int i = size / 2; i > 0; i--) {
		heapAdjust(nums, i,  size);
	}
}

// ������
void heapSort(vector<int>& nums, int size) {
	buildHeap(nums, size); // ����
	for (int i = size; i > 0; i--) {
		swap(nums[i], nums[1]); //�����Ѷ������һ��Ԫ�أ���ÿ�ν�ʣ��Ԫ�ص�����߷ŵ������
		heapAdjust(nums, 1, i - 1); // ���µ����Ѷ��ڵ�Ϊ�󶥶�
	}

}
int main(void) {
	vector<int> res;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> num(0, 10000);
	int size = 100;

	//Ϊ�˷��������0λ�ò���һ���������ݣ�������β������res����Ԫ��
	//res.push_back(0);

	for (int i = 0; i < size; i++) {
		res.push_back(num(gen));
	}
	res.push_back(res.front());



	cout << "unsorted: " << endl;
	for (int i = 1; i <= size; i++) {
		cout << res[i] << ", ";
	}
	cout << endl;

	heapSort(res, size);
	cout << "sorted" << endl;
	for (int i = 1; i <= size; i++) {
		cout << res[i] << ", ";
	}
	cout << endl;

	return 0;
}