#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <thread>
#include <chrono>
#include "visualization.h"
using namespace std;

int main(){

    visualization viz;

    // Example usage: setting points externally
    Eigen::MatrixXf newPoints(1000, 3);
    newPoints.setRandom(); // Random points for demonstration
    viz.points1 = newPoints;

    viz.display();

    glutMainLoop();
    cout << "done" << endl;
    return 0;

}


// // Global visualization instance
// visualization* viz;

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