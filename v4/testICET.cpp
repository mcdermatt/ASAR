#include "icet.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "csv-parser/single_include/csv.hpp"
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>  // Include the algorithm header for std::sort
#include <map>

using namespace Eigen;
using namespace std;

//testing calling ICET from seprate script 
// excuse the mess

// /usr/bin/g++ -fdiagnostics-color=always -g /home/derm/ASAR/v4/testICET.cpp /home/derm/ASAR/v4/icet.cpp -O3 -o /home/derm/ASAR/v4/testICET -I/home/derm/vcpkg/installed/x64-linux/include -L/home/derm/vcpkg/installed/x64-linux/lib -lglfw -lGLEW -lGL -lGLU -lglut npy.hpp -I/home/derm/ASAR/v4 -pthread

int main(int argc, char** argv) {
    
    Eigen::MatrixXf points(250000, 3);  // Declare points as a global variable
    Eigen::MatrixXf points2(250000, 3);  // Declare points as a global variable

    auto before = std::chrono::system_clock::now();
    auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);

    //load point cloud files
    // // Load Ouster Sample Dataset
    // std::string csvFilePath1 = "sample_data/pcap_out_000261.csv";
    // std::string csvFilePath2 = "sample_data/pcap_out_000262.csv";
    // string datasetType = "ouster";
    // // Load generic csv dataset
    // std::string csvFilePath1 = "sample_data/big_curve_scan1.txt";
    // std::string csvFilePath2 = "sample_data/big_curve_scan2.txt";
    std::string csvFilePath1 = "/home/derm/rosbag/desk_test_25.txt";
    std::string csvFilePath2 = "/home/derm/rosbag/desk_test_26.txt";
    string datasetType = "txt";
    points = loadPointCloudCSV(csvFilePath1, datasetType);
    points2 = loadPointCloudCSV(csvFilePath2, datasetType);

    auto after1 = std::chrono::system_clock::now();
    auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
    auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
    std::cout << "loaded point clouds in: " << elapsedTimeMs << " ms" << std::endl;

    auto before2 = std::chrono::system_clock::now();
    auto before2Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(before2);

    //init ICET parameters
    Eigen::VectorXf X0(6);
    X0 << 0., 0, 0., 0., 0., 0.;
    int numBinsPhi = 24;
    int numBinsTheta = 75;
    int n = 25; // min size of the cluster
    float thresh = 0.1; // Jump threshold for beginning and ending radial clusters
    float buff = 0.1; //buffer to add to inner and outer cluster range (helps attract nearby distributions)
    int runlen = 5; //number of iterations
    bool draw = false;

    //run ICET
    Eigen::VectorXf X = icet(points, points2, X0, numBinsPhi, numBinsTheta, n, thresh, buff, runlen, draw);
    std::cout << "X: \n " << X << endl;

    auto after2 = std::chrono::system_clock::now();
    auto after2Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after2);
    auto elapsedTime2Ms = std::chrono::duration_cast<std::chrono::milliseconds>(after2Ms - before2Ms).count();
    std::cout << "ran ICET in: " << elapsedTime2Ms << " ms" << std::endl;


    return 0;
}