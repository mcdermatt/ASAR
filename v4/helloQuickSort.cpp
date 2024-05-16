#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <list>

using namespace std;

int partition(vector<int>& nums, int low, int high){

    //choose the pivot
    int pivot = nums[high];
    int i = low - 1; //index of the smallest element

    for (int j = low; j<high; ++j){
        //if current element is smaller than or equal to pivot
        if (nums[j] < pivot){
            ++i;
            swap(nums[i], nums[j]);
        }
    }

    swap(nums[i+1], nums[high]);
    return i + 1; // return pivot index

}

void quicksort(vector<int>& nums, int low, int high){
    if (low< high) {
        int pivotIndex = partition(nums, low, high);

        cout << " here we go again, with pivot index: " << pivotIndex << endl;
        for (auto n:nums){
            cout << n;
        }
        cout << endl;

        //recursively sort elements before and after the pivot
        quicksort(nums, low, pivotIndex-1);
        quicksort(nums, pivotIndex+1, high);
    }
}

int main(){
    vector<int> nums = {4,5,2,2,9,1,3,4};

    cout << "OG nums: ";
    for (auto n:nums){
        cout << n;
    }

    cout  << endl;

    quicksort(nums, 0, nums.size()-1);



}