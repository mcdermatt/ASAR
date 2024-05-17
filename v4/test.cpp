#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <thread>
#include <chrono>
#include "visualization.h"
using namespace std;

int main(){

    visualization viz;

    //set points externally
    Eigen::MatrixXf new1(1000, 3);
    new1.setRandom();
    viz.points1 = new1;
    Eigen::MatrixXf new2(1000, 3);
    new2.setRandom();
    viz.points2 = new2;

    //set frustum externally using clusterBounds
    Eigen::MatrixXf cb(10,6);
    for (int i = 0; i < 10; i++){
        cb.row(i) << i * M_PI/5, (i + 1)*M_PI/5 , 7*M_PI/16, M_PI/2, 2 + i/3, 3+i/3;
    }
    viz.clusterBounds = cb;

    //set covariance ellipsoids
    vector<Eigen::Vector3f> e1mean;  
    Eigen::Vector3f mu1 = {0.,1.,2.};
    e1mean.push_back(mu1);
    vector<Eigen::Matrix3f> e1cov;
    Eigen::Matrix3f cov1 = Eigen::Matrix3f::Identity();
    e1cov.push_back(cov1);
    vector<float> e1alpha = {0.5};
    viz.ellipsoid1Means = e1mean;
    viz.ellipsoid1Covariances = e1cov;
    viz.ellipsoid1Alphas = e1alpha;

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