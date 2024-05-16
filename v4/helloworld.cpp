#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <list>

using namespace std;

struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
     };

int main()
{

    // vector<int> nums1 {1,2,3, 0, 0, 0};
    // vector<int> nums2 {1,3,8};

    // nums1.erase(nums1.begin() + 3, nums1.end());
    
    // for (int i = 0; i < 3; i++){
    //     nums1.push_back(nums2[i]);
    // }

    // std::sort(nums1.begin(),nums1.end());

    // for (int j = 0; j < nums1.size(); j++){
    //     cout << nums1[j];
    // }

    // int a = 1;
    // int b = 2;
    // auto c = a ^ b;
 
    // cout << c << endl;


    // unordered_map<int, list<int>> myMap = {
    //     {1, {1,2,3}},
    //     {2, {4,5,6}}
    // };

    // for(auto node: myMap[1]){
    //     cout << node << endl;
    // }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // string pattern = "abba";
    // string s = "beep boop boop beep";

    // unordered_map<string, string> myMap;
    // int pattern_count = 0;
    // string temp = "";

    // int idx = 0;
    // for (char c:s){

    //     //word break
    //     if ((c == ' ') || (idx == s.size()))  {
    //         myMap[temp].push_back(pattern_count);
    //         cout << temp << endl;
    //         temp = "";
    //         pattern_count ++;
    //     }
    //     else{
    //         temp.push_back(c);
    //     }
    //     idx ++;

    // }

    // cout << "---------------" << endl;

    // for (auto m = myMap.begin(); m != myMap.end(); ++m){

    //     cout << m->first << endl;
    //     cout << m->second[0] << endl;
    // }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // int len = 5;
    // vector<vector<int>> ans = {{1},
    //                            {1,1}};


    // for (int row = 2; row< len; row++){
    //     vector<int> new_row = {1};
    //     for (int i=0; i < row; i ++){
    //         new_row.push_back(ans[row-1][i] + ans[row-1][i+1]);
    //     }
    //     ans.push_back(new_row);
    // }

    // //draw triangle
    // for (int i = 0; i < ans.size(); i++){
    //     for (int j=0; j < ans[i].size(); j++){
    //         cout << ans[i][j];
    //     }
    //     cout << endl;
    // }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


}