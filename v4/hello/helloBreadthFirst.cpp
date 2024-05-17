#include <iostream>
#include <queue>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int minDepth(TreeNode* root) {
    if (!root) return 0;

    // Create a queue for level order traversal
    queue<TreeNode*> q;
    q.push(root);

    // Initialize depth to 1
    int depth = 1;

    // Level order traversal
    while (!q.empty()) {
        int size = q.size();
        // Traverse all the nodes at the current level
        for (int i = 0; i < size; ++i) {
            TreeNode* node = q.front();
            q.pop();

            // If the current node is a leaf, return the depth
            if (!node->left && !node->right) return depth;

            // Add left and right children to the queue if they exist
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        // Increment depth after traversing all nodes at the current level
        ++depth;
    }
    return depth;
}

int main() {
    // Construct the binary tree
    TreeNode* root1 = new TreeNode(3);
    root1->left = new TreeNode(9);
    root1->right = new TreeNode(20);
    root1->right->left = new TreeNode(15);
    root1->right->right = new TreeNode(7);

    TreeNode* root2 = new TreeNode(2);
    root2->right = new TreeNode(3);
    root2->right->right = new TreeNode(4);
    root2->right->right->right = new TreeNode(5);
    root2->right->right->right->right = new TreeNode(6);

    cout << minDepth(root1) << endl; // Output: 2
    cout << minDepth(root2) << endl; // Output: 5

    // Deallocate memory to avoid memory leaks
    delete root1->left;
    delete root1->right->left;
    delete root1->right->right;
    delete root1->right;
    delete root1;

    delete root2->right->right->right->right;
    delete root2->right->right->right;
    delete root2->right->right;
    delete root2->right;
    delete root2;

    queue<string> Q = {};
    Q.push("a");
    Q.push("b");
    Q.push("c");
    Q.pop();
    while (!Q.empty()){
        cout << Q.front() << endl;
        Q.pop();
    }

    return 0;
}
