#include <iostream>
// #include <eigen3/Eigen/Dense> //works by default
#include <unsupported/Eigen/CXX11/Tensor> // works with $>> g++ -o eigenTest eigenTest.cpp -I/usr/include/eigen3
// #include <Eigen/unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense> //needed to link file path to get this to work
#include <ctime> 
#include <random>

using Eigen::MatrixXd;
 
int main()
{
//   MatrixXd m(2,2);
//   m(0,0) = 3;
//   m(1,0) = 2.5;
//   m(0,1) = -1;
//   m(1,1) = m(1,0) + m(0,1);
//   std::cout << m << std::endl;

//   double HI = 12345.67; // set HI and LO according to your problem.
//   double LO = 879.01;
//   double range= HI-LO;
//   MatrixXd m = MatrixXd::Random(3,3); // 3x3 Matrix filled with random numbers between (-1,1)
//   m = (m + MatrixXd::Constant(3,3,1.))*range/2.; // add 1 to the matrix to have values between 0 and 2; multiply with range/2
//   m = (m + MatrixXd::Constant(3,3,LO)); //set LO as the lower bound (offset)
//   std::cout << "m =\n" << m << std::endl;

//   int t = rand(); // create random number at COMPILE TIME
// // create random number at RUNTIME
//   srand((unsigned) time(0));
//   int t = 1 + (rand() % 6); 
//   std::cout << "t =\n" << t << std::endl;

// // Create a 3x3x3 tensor
//     Eigen::Tensor<double, 3> tensor(3, 3, 3);

//     // Fill the tensor with values
//     tensor.setValues({{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                       {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
//                       {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}}});

//     // Set a specific element (e.g., element at row 1, column 2) to a new value
//     tensor.coeffRef(0,1, 2) = 42.0;

//     // Access and print values
//     for (int i = 0; i < tensor.dimension(0); ++i) {
//         for (int j = 0; j < tensor.dimension(1); ++j) {
//             for (int k = 0; k < tensor.dimension(2); ++k) {
//                 // tensor.coeffRef(0,i, j) = 42.0;
//                 std::cout << "tensor(" << i << ", " << j << ", " << k << ") = " << tensor(i, j, k) << "\n";
//             }
//         }
//     }


    // create a gaussian(ish) distribution
    // Eigen::MatrixXf vec = Eigen::MatrixXf::Random(10, 3);
    // std::cout << "vec =\n" << vec << std::endl;
    // Set up random number generator with normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, 1.0f); // mean = 0, stddev = 1

    // Create a 10 by 3 matrix with Gaussian-distributed x, y, and z
    Eigen::MatrixXf gaussianVector(10, 3);

    for (int i = 0; i < 10; ++i) {
        gaussianVector(i, 0) = distribution(gen);
        gaussianVector(i, 1) = distribution(gen);
        gaussianVector(i, 2) = distribution(gen);
    }

    // Display the generated vector
    std::cout << "Gaussian Vector:\n" << gaussianVector << "\n";

    return 0;

}