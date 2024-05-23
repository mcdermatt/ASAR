#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <thread>
#include <chrono>
#include "visualization.h"
#include "utils.h"
#include "icet.h"
using namespace std;

//file for testing running updated ICET with visualization

int main(){

    visualization viz;

    // Load Ouster Sample Dataset
    std::string csvFilePath1 = "sample_data/pcap_out_000261.csv";
    std::string csvFilePath2 = "sample_data/pcap_out_000262.csv";
    string datasetType = "ouster";

    // // Load Matt's apartment dataset
    // std::string csvFilePath1 = "/home/derm/rosbag/desk_test_20.txt";
    // std::string csvFilePath2 = "/home/derm/rosbag/desk_test_21.txt";
    // string datasetType = "txt";

    Eigen::MatrixXf new1 = utils::loadPointCloudCSV(csvFilePath1, datasetType);
    Eigen::MatrixXf new2 = utils::loadPointCloudCSV(csvFilePath2, datasetType);

    int rl = 5;
    ICET it(new1, new2, rl);

    viz.points1 = it.points1;
    viz.points2 = it.points2;
    viz.clusterBounds = it.clusterBounds;

    // cout << "filled ells inside main:  "<< it.ellipsoid1Means.size() << endl;
    // cout << "occupied count: " << it.occupiedCount << endl;

    // //set covariance ellipsoids
    viz.ellipsoid1Means = it.ellipsoid1Means;
    viz.ellipsoid1Covariances = it.ellipsoid1Covariances;
    viz.ellipsoid1Alphas = it.ellipsoid1Alphas;

    viz.ellipsoid2Means = it.ellipsoid2Means;
    viz.ellipsoid2Covariances = it.ellipsoid2Covariances;
    viz.ellipsoid2Alphas = it.ellipsoid2Alphas;

    viz.display();
    glutMainLoop();
    return 0;
}


// TODO: create two threads, call ICET with one, enter the glutMainLoop() with the other to watch algorithm converge

//// update points while viz is running
// visualization* viz; // Global visualization instance

// // Function to update points
// void updatePoints() {
//     while (true) {
//         Eigen::MatrixXf newPoints(100, 3);
//         newPoints.setRandom(); // Generate new random points
//         viz->points1 = newPoints;
//         glutPostRedisplay(); // Request to update the display

//         std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Simulate data update interval
//     }
// }

// int main() {
//     viz = new visualization;
//     std::cout << "done" << std::endl;

//     // Initial points setup
//     Eigen::MatrixXf initialPoints(100, 3);
//     initialPoints.setRandom(); // Random points for demonstration
//     viz->points1 = initialPoints;

//     // Start a thread to update points
//     std::thread updateThread(updatePoints);
//     updateThread.detach(); // Detach the thread to run independently

//     glutMainLoop();
//     return 0;
// }