#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

// 堆调整
void heapAdjust(vector<int>& nums, int i, int size) {
	if (i > size / 2) return; // 如果i是叶节点则不用进行调整
	
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
		heapAdjust(nums, max, size); //避免调整之后以max为父节点的子树不是大顶堆
	}
 
}

// 建堆
void buildHeap(vector<int>& nums, int size) {
	for (int i = size / 2; i > 0; i--) {
		heapAdjust(nums, i,  size);
	}
}

// 堆排序
void heapSort(vector<int>& nums, int size) {
	buildHeap(nums, size); // 建堆
	for (int i = size; i > 0; i--) {
		swap(nums[i], nums[1]); //交换堆顶和最后一个元素，即每次将剩余元素的最大者放到最后面
		heapAdjust(nums, 1, i - 1); // 重新调整堆顶节点为大顶堆
	}

}
int main(void) {
	vector<int> res;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> num(0, 10000);
	int size = 100;

	//为了方便计数，0位置插入一个垃圾数据；或者在尾部插入res的首元素
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