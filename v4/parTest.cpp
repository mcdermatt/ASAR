#include <execution>
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Create a vector of integers
    std::vector<int> vec = {9, 3, 1, 7, 5, 8, 2, 4, 6};

    // Print the original vector
    std::cout << "Original vector:";
    for (int num : vec) {
        std::cout << " " << num;
    }
    std::cout << std::endl;

    // Sort the vector in parallel using a lambda function
    std::sort(std::execution::par, vec.begin(), vec.end(), [](int a, int b) {
        return a < b;
    });

    // Print the sorted vector
    std::cout << "Sorted vector:";
    for (int num : vec) {
        std::cout << " " << num;
    }
    std::cout << std::endl;

    return 0;
}
