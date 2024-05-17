#ifndef VISUALIZATION_H
#define VISUALIZATION_H

// Include necessary OpenGL headers
#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
// #include <random>
// #include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <string>
// #include <glm/glm.hpp>
// #include "csv-parser/single_include/csv.hpp"
// #include <fstream>
// #include <cmath>
// #include <limits>
// #include <algorithm> 
// #include <map>
// #include <execution>
// #include "ThreadPool.h"

class visualization {
public:
    visualization(); // Constructor // int argc, char** argv
    ~visualization(); // Destructor

    void initializeOpenGL(); //int argc, char** argv
    void render();
    static void display();
    static void reshape(int w, int h);
    static void specialKeys(int key, int x, int y);
    static void mouse(int button, int state, int x, int y);
    static void motion(int x, int y);
    static void keyboard(unsigned char key, int x, int y);    

    Eigen::MatrixXf points1;

private:
    void handleReshape(int w, int h);
    void handleSpecialKeys(int key, int x, int y);
    void handleMouse(int button, int state, int x, int y);
    void handleMotion(int x, int y);
    void handleKeyboard(unsigned char key, int x, int y);

    GLdouble cameraDistance;
    GLdouble cameraAngleX;
    GLdouble cameraAngleY;
    int lastMouseX;
    int lastMouseY;

    void drawPoints1(); // Dummy function for drawing points
    void drawPoints2(); // Dummy function for drawing points
    void drawFrustum(float x, float y, float z, float w, float h, float d); // Dummy function for drawing frustum
    void drawEllipsoid(const std::array<float, 3>& mean, const std::array<float, 9>& covariance, float alpha, const std::array<float, 3>& color); // Dummy function for drawing ellipsoid

    std::vector<std::array<float, 3>> ellipsoid1Means;
    std::vector<std::array<float, 9>> ellipsoid1Covariances;
    std::vector<float> ellipsoid1Alphas;

    std::vector<std::array<float, 3>> ellipsoid2Means;
    std::vector<std::array<float, 9>> ellipsoid2Covariances;
    std::vector<float> ellipsoid2Alphas;

    Eigen::MatrixXf clusterBounds; // Dummy variable for cluster bounds

};

#endif // VISUALIZATION_HPP
