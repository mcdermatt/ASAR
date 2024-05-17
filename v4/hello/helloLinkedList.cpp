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

void printLinkedList(ListNode* head) {
    ListNode* current = head;
    while (current != nullptr) {
        std::cout << "New node value: " << current->val;
        std::cout << "   Next: " << current->next << std::endl;     
        current = current->next;
    }
    std::cout << std::endl;
}

ListNode* createLinkedListFromVector(const std::vector<int>& vec) {
    ListNode* head = nullptr;
    ListNode* tail = nullptr;

    for (int val : vec) {
        ListNode* newNode = new ListNode(val);
        if (!head) {
            head = newNode;
            tail = newNode;
        } else {
            tail->next = newNode;
            tail = newNode;
        }
    }

    return head;
}

int main()
{
    vector<int> vec = {1, 2, 3, 4, 5};

    ListNode* head = createLinkedListFromVector(vec);
    // printLinkedList(head);
    // auto test = head->next->next;
    // printLinkedList(test); 

    vector<int> valsA = {};
    vector<ListNode*> nextA = {}; 
    
    ListNode* curr = head;
    while (curr != nullptr){
        cout<< curr->val << endl;
        valsA.push_back(curr->val);
        nextA.push_back(curr->next);
        curr = curr->next;
    }

    for (auto i:nextA){
        cout << i << endl;
    }

}