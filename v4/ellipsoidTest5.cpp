#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
// #include "npy.hpp"
#include "csv-parser/single_include/csv.hpp"
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>  // Include the algorithm header for std::sort
#include <map>

using namespace Eigen;
using namespace std;

// script for drawing spherical voxels on real point cloud data
// works but takes > 400 Ms to fit the clusters (waaaay too slow)

GLuint pointsVBO, ellipsoidVBO, pointsVAO, ellipsoidVAO;

GLdouble cameraDistance = 10.0;
GLdouble cameraAngleX = 45.0;
GLdouble cameraAngleY = -45.0;

GLdouble lastMouseX = 0.0;
GLdouble lastMouseY = 0.0;

Eigen::MatrixXf points(250000, 3);  // Declare points as a global variable
Eigen::MatrixXf points2(250000, 3);  // Declare points as a global variable
Eigen::MatrixXf testPoints(250000, 3); //for debug

std::vector<Eigen::Vector3f> ellipsoid1Means;
std::vector<Eigen::Matrix3f> ellipsoid1Covariances;
std::vector<float> ellipsoid1Alphas;
std::vector<Eigen::Vector3f> ellipsoid2Means;
std::vector<Eigen::Matrix3f> ellipsoid2Covariances;
std::vector<float> ellipsoid2Alphas;

Eigen::MatrixXf frustumVertices(8, 3);
GLuint frustumVBO, frustumVAO;

GLfloat azimuthalMin = 0.0;
GLfloat azimuthalMax = 2.0 * M_PI;
GLfloat elevationMin = -M_PI / 4.0;
GLfloat elevationMax = M_PI / 4.0;
GLfloat innerDistance = 5.0;
GLfloat outerDistance = 10.0;

//set cluster bounds as [n, 6] matrix
Eigen::MatrixXd clusterBounds(10000,6);

//init structure to store covaraince matrices
// Type definition for the covariance matrix
using CovarianceMatrix = Matrix<float, 3, 3>;
// Type definition for the data structure
using CovarianceMap = map<int, map<int, CovarianceMatrix>>;
// using CovarianceMap = map<int, map<int, Matrix<float, 3, 3>>>;

// type definition for means vectors
using MeanMap = map<int, map<int, Vector3f>>;

void createFrustumVBO(GLuint& vbo, GLenum target, const Eigen::MatrixXf& data) {
    glGenBuffers(1, &vbo);
    glBindBuffer(target, vbo);
    glBufferData(target, sizeof(float) * data.size(), data.data(), GL_STATIC_DRAW);
    glBindBuffer(target, 0);
}

void createFrustumVAO(GLuint& vao, GLuint vbo, int vertexSize, GLuint shaderAttributeIndex, GLsizei dataSize) {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(shaderAttributeIndex, vertexSize, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(shaderAttributeIndex);
    if (dataSize > 0) {
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * dataSize, nullptr, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void initFrustumVBOAndVAO() {
    Eigen::MatrixXf frustumVertices(8, 3);
    // Define frustum vertices here (modify as needed)
    frustumVertices <<
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f;

    createFrustumVBO(frustumVBO, GL_ARRAY_BUFFER, frustumVertices);
    createFrustumVAO(frustumVAO, frustumVBO, 3, 2, frustumVertices.rows());
}

void drawCroppedSphere(GLfloat radius, GLfloat azimuthalMin, GLfloat azimuthalMax,
                       GLfloat elevationMin, GLfloat elevationMax, GLint slices, GLint stacks) {
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    for (GLint i = 0; i < slices; ++i) {
        glBegin(GL_TRIANGLE_STRIP);
        for (GLint j = 0; j <= stacks; ++j) {
            // Calculate the azimuthal and elevation angles in radians
            GLfloat azimuthalAngle1 = azimuthalMin + (azimuthalMax - azimuthalMin) * i / slices;
            GLfloat elevationAngle1 = elevationMin + (elevationMax - elevationMin) * j / stacks;

            GLfloat azimuthalAngle2 = azimuthalMin + (azimuthalMax - azimuthalMin) * (i + 1) / slices;
            GLfloat elevationAngle2 = elevationMin + (elevationMax - elevationMin) * j / stacks;

            // Calculate the coordinates on the sphere
            GLfloat x1 = radius * sin(elevationAngle1) * cos(azimuthalAngle1);
            GLfloat y1 = radius * sin(elevationAngle1) * sin(azimuthalAngle1);
            GLfloat z1 = radius * cos(elevationAngle1);

            GLfloat x2 = radius * sin(elevationAngle2) * cos(azimuthalAngle2);
            GLfloat y2 = radius * sin(elevationAngle2) * sin(azimuthalAngle2);
            GLfloat z2 = radius * cos(elevationAngle2);

            // Draw the vertices
            glVertex3f(x1, y1, z1);
            glVertex3f(x2, y2, z2);
        }
        glEnd();
    }
    // glDisable(GL_BLEND);
    // glDepthMask(GL_TRUE);
    // glDisable(GL_DEPTH_TEST);
}


void drawFrustum(GLfloat azimuthalMin, GLfloat azimuthalMax, GLfloat elevationMin, GLfloat elevationMax,
                  GLfloat innerDistance, GLfloat outerDistance) {
    glBegin(GL_LINES);

    // Define the eight points
    GLfloat points[8][3];

    // Near surface
    points[0][0] = innerDistance * std::sin(elevationMin) * std::cos(azimuthalMin);
    points[0][1] = innerDistance * std::sin(elevationMin) * std::sin(azimuthalMin);
    points[0][2] = innerDistance * std::cos(elevationMin);

    points[1][0] = innerDistance * std::sin(elevationMin) * std::cos(azimuthalMax);
    points[1][1] = innerDistance * std::sin(elevationMin) * std::sin(azimuthalMax);
    points[1][2] = innerDistance * std::cos(elevationMin);

    points[2][0] = innerDistance * std::sin(elevationMax) * std::cos(azimuthalMax);
    points[2][1] = innerDistance * std::sin(elevationMax) * std::sin(azimuthalMax);
    points[2][2] = innerDistance * std::cos(elevationMax);

    points[3][0] = innerDistance * std::sin(elevationMax) * std::cos(azimuthalMin);
    points[3][1] = innerDistance * std::sin(elevationMax) * std::sin(azimuthalMin);
    points[3][2] = innerDistance * std::cos(elevationMax);

    // Far surface
    points[4][0] = outerDistance * std::sin(elevationMin) * std::cos(azimuthalMin);
    points[4][1] = outerDistance * std::sin(elevationMin) * std::sin(azimuthalMin);
    points[4][2] = outerDistance * std::cos(elevationMin);

    points[5][0] = outerDistance * std::sin(elevationMin) * std::cos(azimuthalMax);
    points[5][1] = outerDistance * std::sin(elevationMin) * std::sin(azimuthalMax);
    points[5][2] = outerDistance * std::cos(elevationMin);

    points[6][0] = outerDistance * std::sin(elevationMax) * std::cos(azimuthalMax);
    points[6][1] = outerDistance * std::sin(elevationMax) * std::sin(azimuthalMax);
    points[6][2] = outerDistance * std::cos(elevationMax);

    points[7][0] = outerDistance * std::sin(elevationMax) * std::cos(azimuthalMin);
    points[7][1] = outerDistance * std::sin(elevationMax) * std::sin(azimuthalMin);
    points[7][2] = outerDistance * std::cos(elevationMax);

    // Draw lines connecting the points
    for (int i = 0; i < 4; ++i) {
        // // Near surface
        // glVertex3fv(points[i]);
        // glVertex3fv(points[(i + 1) % 4]);

        // // Far surface
        // glVertex3fv(points[i + 4]);
        // glVertex3fv(points[(i + 1) % 4 + 4]);

        // Connecting lines
        glVertex3fv(points[i]);
        glVertex3fv(points[i + 4]);
    }

    // // X-axis (Red)
    // glColor3f(1.0, 0.0, 0.0);
    // glVertex3f(0.0, 0.0, 0.0);
    // glVertex3f(10.0, 0.0, 0.0);

    // // Y-axis (Green)
    // glColor3f(0.0, 1.0, 0.0);
    // glVertex3f(0.0, 0.0, 0.0);
    // glVertex3f(0.0, 10.0, 0.0);

    // // Z-axis (Blue)
    // glColor3f(0.0, 0.0, 1.0);
    // glVertex3f(0.0, 0.0, 0.0);
    // glVertex3f(0.0, 0.0, 10.0);

    glEnd();

    // // Draw partial sphere on the inner and outer surfaces
    glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE);
    glDepthMask(GL_FALSE);

    // // Enable polygon smoothing for antialiasing
    // glEnable(GL_POLYGON_SMOOTH);
    // glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glColor4f(0.8, 0.8, 1.0, 0.7); 
    drawCroppedSphere(innerDistance, azimuthalMin, azimuthalMax, elevationMin, elevationMax, 12, 12);
    glColor4f(0.8, 0.8, 1.0, 0.7);  
    drawCroppedSphere(outerDistance, azimuthalMin, azimuthalMax, elevationMin, elevationMax, 12, 12);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);  // Disable depth testing after drawing

 
}

void createVBO(GLuint& vbo, GLenum target, const Eigen::MatrixXf& data) {
    glGenBuffers(1, &vbo);
    glBindBuffer(target, vbo);
    glBufferData(target, sizeof(float) * data.size(), data.data(), GL_STATIC_DRAW);
    glBindBuffer(target, 0);
}

void createVAO(GLuint& vao, GLuint vbo, int vertexSize, GLuint shaderAttributeIndex, GLsizei dataSize = 0) {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(shaderAttributeIndex, vertexSize, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(shaderAttributeIndex);
    if (dataSize > 0) {
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * dataSize, nullptr, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// void initVBOsAndVAOs() {
//     // Create VBO and VAO for points
//     // //test ~~~~~~~~~~~~
//     // // Generate Vertex Buffer Object (VBO)
//     // glGenBuffers(1, &pointsVBO);
//     // glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
//     // glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), points.data(), GL_STATIC_DRAW);

//     // // Generate Vertex Array Object (VAO)
//     // glGenVertexArrays(1, &pointsVAO);
//     // glBindVertexArray(pointsVAO);

//     // // Specify the layout of the vertex data
//     // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
//     // glEnableVertexAttribArray(0);

//     // // Unbind VAO and VBO
//     // glBindVertexArray(0);
//     // glBindBuffer(GL_ARRAY_BUFFER, 0);
//     // // ~~~~~~~~~~~~~~~~

//     // OLD-- using same function as ellipsoids
//     createVBO(pointsVBO, GL_ARRAY_BUFFER, points);
//     createVAO(pointsVAO, pointsVBO, 3, 0, points.rows());  // Pass the size of points matrix

//     // Create VBO and VAO for ellipsoids
//     Eigen::MatrixXf ellipsoidVertices(100, 3);  // 10x10 = 100 
//     for (int i = 0; i < 100; ++i) {
//         float theta = 2.0 * M_PI * i / 100;
//         ellipsoidVertices(i, 0) = cos(theta);
//         ellipsoidVertices(i, 1) = sin(theta);
//         ellipsoidVertices(i, 2) = 0.0f;
//     }

//     createVBO(ellipsoidVBO, GL_ARRAY_BUFFER, ellipsoidVertices);
//     createVAO(ellipsoidVAO, ellipsoidVBO, 3, 1, ellipsoidVertices.rows());  // Pass the size of ellipsoidVertices matrix

//     // Initialize VBO and VAO for frustum
//     initFrustumVBOAndVAO();

// }


Eigen::MatrixXf generateEigenNormal(int rows, int cols, float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(mean, stddev);

    Eigen::MatrixXf result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = distribution(gen);
        }
    }

    return result;
}

Eigen::MatrixXf generateEigenCovariance(int rows, const Eigen::Vector3f& mean, const Eigen::Matrix3f& covariance) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Perform Cholesky decomposition on the covariance matrix
    Eigen::LLT<Eigen::Matrix3f> lltOfCovariance(covariance);
    Eigen::Matrix3f lowerTriangular = lltOfCovariance.matrixL();

    // Generate random points from a standard normal distribution
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    Eigen::MatrixXf randomMatrix(rows, 3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < 3; ++j) {
            randomMatrix(i, j) = distribution(gen);
        }
    }

    // Transform the standard normal points to follow the specified distribution
    Eigen::MatrixXf points = randomMatrix * lowerTriangular.transpose();

    // Add the mean vector to the generated points
    points.rowwise() += mean.transpose();

    return points;
}

void drawEllipsoid(const Eigen::Vector3f& mean, const Eigen::Matrix3f& covariance, GLfloat alpha, const std::array<float, 3>& color) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
    Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
    Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors().real();

    glPushMatrix();
    glTranslatef(mean(0), mean(1), mean(2));

    // Apply rotation based on eigenvectors
    Eigen::Matrix4f rotationMatrix;
    rotationMatrix << eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2), 0,
                      eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2), 0,
                      eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2), 0,
                      0, 0, 0, 1;
    glMultMatrixf(rotationMatrix.data());

    // Apply non-uniform scaling based on eigenvalues
    glScalef(2 * sqrt(eigenvalues(0)), 2 * sqrt(eigenvalues(1)), 2 * sqrt(eigenvalues(2)));

    // glColor4f(0.0, 0.0, 1.0, alpha);
    glColor4f(color[0], color[1], color[2], alpha);
    // glutSolidSphere(1.0, 100, 100); //way too high res
    glutSolidSphere(1.0, 10, 10);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glPopMatrix();
}

// // modern openGL
// void drawPoints() {
//     // std::cout << "Drawing Points. Size: " << points.rows() << std::endl;
//     glBindVertexArray(pointsVAO);
//     // glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
//     glPointSize(5.0f);  // Increase point size
//     glColor3f(1.0, 1.0, 1.0);

//     glDrawArrays(GL_POINTS, 0, points.rows());
//     // glBindBuffer(GL_ARRAY_BUFFER, 0);
//     glBindVertexArray(0);
// }

// TODO: stop using this
void drawPoints1() {
    // std::cout << "Drawing Points. Size: " << points.rows() << std::endl;
    glPointSize(3.0f);
    // glColor3f(1.0, 1.0, 1.0);  // White color
    glColor3f(0.5, 0.5, 1.0);  // blue

    glBegin(GL_POINTS);
    for (int i = 0; i < points.rows(); ++i) {
        glVertex3f(points(i, 0), points(i, 1), points(i, 2));
    }
    glEnd();
}

void drawPoints2() {
    // std::cout << "Drawing Points. Size: " << points.rows() << std::endl;
    glPointSize(3.0f);
    glColor3f(1., 0.5, 0.5);  // red
    glBegin(GL_POINTS);
    for (int i = 0; i < points2.rows(); ++i) {
        glVertex3f(points2(i, 0), points2(i, 1), points2(i, 2));
    }
    glEnd();
}

void drawTestPoints() {
    // std::cout << "Drawing Points. Size: " << points.rows() << std::endl;
    glPointSize(5.0f);
    glColor3f(1.0, 0.0, 0.0);  // White color

    glBegin(GL_POINTS);
    for (int i = 0; i < testPoints.rows(); ++i) {
        glVertex3f(testPoints(i, 0), testPoints(i, 1), testPoints(i, 2));
    }
    glEnd();
}


void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<GLdouble>(w) / h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
}

void specialKeys(int key, int x, int y) {
    if (key == GLUT_KEY_UP) {
        cameraDistance -= 0.5;
    }
    else if (key == GLUT_KEY_DOWN) {
        cameraDistance += 0.5;
    }

    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
    }
}

void motion(int x, int y) {
    cameraAngleY += (x - lastMouseX) * 0.1;
    cameraAngleX += (y - lastMouseY) * 0.1;

    lastMouseX = x;
    lastMouseY = y;

    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 27) {
        exit(0);
    }
}

// Function to convert Cartesian coordinates to spherical coordinates
MatrixXf cartesianToSpherical(const MatrixXf& cartesianPoints) {
    // Ensure that the input matrix has 3 columns (X, Y, Z coordinates)
    assert(cartesianPoints.cols() == 3);

    // Extract X, Y, Z columns
    VectorXf x = cartesianPoints.col(0);
    VectorXf y = cartesianPoints.col(1);
    VectorXf z = cartesianPoints.col(2);

    // Compute radius (r)
    VectorXf r = cartesianPoints.rowwise().norm();

    // Compute azimuthal angle (theta)
    VectorXf theta = VectorXf::Zero(cartesianPoints.rows());
    for (int i = 0; i < cartesianPoints.rows(); ++i) {
        theta(i) = std::atan2(y(i), x(i));
        if (theta(i) < 0.0) {
            theta(i) += 2.0 * M_PI;
        }
    }

    // Compute elevation angle (phi) as the complement from the xy-plane
    VectorXf phi = VectorXf::Zero(cartesianPoints.rows());
    for (int i = 0; i < cartesianPoints.rows(); ++i) {
        phi(i) = std::acos(z(i) / r(i));
        // std::cout << "phi(i): \n" << phi(i) << "\n";
    }

    // Combine r, theta, phi into a new matrix
    MatrixXf sphericalPoints(cartesianPoints.rows(), 3);
    sphericalPoints << r, theta, phi;

    // Replace NaN or -NaN values with [0, 0, 0]
    for (int i = 0; i < sphericalPoints.rows(); ++i) {
        for (int j = 0; j < sphericalPoints.cols(); ++j) {
            if (std::isnan(sphericalPoints(i, j))) {
                sphericalPoints(i, j) = 1000.0;
            }
        }
    }

    return sphericalPoints;
}


Eigen::MatrixXf sphericalToCartesian(const Eigen::MatrixXf& sphericalPoints) {
    // Ensure that the input matrix has 3 columns (r, theta, phi)
    assert(sphericalPoints.cols() == 3);

    // Extract r, theta, phi columns
    Eigen::VectorXf r = sphericalPoints.col(0);
    Eigen::VectorXf theta = sphericalPoints.col(1);
    Eigen::VectorXf phi = sphericalPoints.col(2);

    // Convert spherical coordinates to Cartesian coordinates
    Eigen::MatrixXf cartesianPoints(sphericalPoints.rows(), 3);

    for (int i = 0; i < sphericalPoints.rows(); ++i) {
        float x = r(i) * sin(phi(i)) * cos(theta(i));
        float y = r(i) * sin(phi(i)) * sin(theta(i));
        float z = r(i) * cos(phi(i));

        cartesianPoints.row(i) << x, y, z;
    }

    return cartesianPoints;
}

pair<float, float> findCluster(const MatrixXf& sphericalCoords, int n, float thresh, float buff) {
    int numPoints = sphericalCoords.rows();

    float innerDistance = 0.0;
    float outerDistance = 0.0;
    vector<Vector3f> localPoints;

    for (int i = 0; i < numPoints; i++) {
        Vector3f point = sphericalCoords.row(i);

        // Check if the point is within the threshold of the last point
        if (!localPoints.empty() && std::abs(localPoints.back()(0) - point(0)) <= thresh) {
            // Add the point to the current cluster
            localPoints.push_back(point);
        } else {
            // Check if the cluster is sufficiently large
            if (localPoints.size() >= n) {
                // Found a sufficiently large cluster
                innerDistance = localPoints.front()(0) - buff;
                outerDistance = localPoints.back()(0) + buff;
                // cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
                return {innerDistance, outerDistance};
            } else {
                // Reset the cluster if it's not large enough
                localPoints.clear();
                // Add the current point to start a new cluster
                localPoints.push_back(point);
            }
        }
    }
    // Check for the last cluster at the end of the loop
    if (localPoints.size() >= n) {
        // innerDistance = localPoints.front()(0);
        // outerDistance = localPoints.back()(0);
        if (localPoints.front()(0) !=0){
            innerDistance = localPoints.front()(0) - buff;
            outerDistance = localPoints.back()(0) + buff;
            // cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
            return {innerDistance, outerDistance};
        }
        else{
            return {0.0, 0.0};
        }
    }

    return {innerDistance, outerDistance};
}


vector<vector<vector<int>>> sortSphericalCoordinates(const MatrixXf& sphericalCoords, int numBinsTheta, int numBinsPhi) {
    // Create a 2D vector of vectors to store point indices in each bin
    vector<vector<vector<int>>> pointIndices(numBinsTheta, vector<vector<int>>(numBinsPhi));

    // Iterate through each spherical coordinate
    for (int i = 0; i < sphericalCoords.rows(); ++i) {
        // Extract phi and theta values
        float theta = sphericalCoords(i, 1);
        float phi = sphericalCoords(i, 2);

        // Calculate bin indices
        int binTheta = static_cast<int>((theta / (2 * M_PI)) * numBinsTheta) % numBinsTheta;
        int binPhi = static_cast<int>((phi / M_PI) * numBinsPhi) % numBinsPhi;

        // Store the point index in the corresponding bin
        pointIndices[binTheta][binPhi].push_back(i);
    }

    // Return the vector of point indices
    return pointIndices;
}

MatrixXf filterPointsInsideCluster(const MatrixXf& selectedPoints, const MatrixXd& clusterBounds) {
    int numPoints = selectedPoints.rows();
    int numClusters = clusterBounds.rows();

    MatrixXf filteredPoints(numPoints, 3);
    int filteredRowCount = 0;

    for (int i = 0; i < numClusters; i++) {
        float azimMin = clusterBounds(i, 0);
        float azimMax = clusterBounds(i, 1);
        float elevMin = clusterBounds(i, 2);
        float elevMax = clusterBounds(i, 3);
        float innerDistance = clusterBounds(i, 4);
        float outerDistance = clusterBounds(i, 5);

        for (int j = 0; j < numPoints; j++) {
            float azim = selectedPoints(j, 1);
            float elev = selectedPoints(j, 2);
            float r = selectedPoints(j, 0);

            // Check if the point is within the cluster bounds
            if (azim >= azimMin && azim <= azimMax &&
                elev >= elevMin && elev <= elevMax &&
                r >= innerDistance && r <= outerDistance) {
                // Add the point to the filteredPoints matrix
                filteredPoints.row(filteredRowCount++) = selectedPoints.row(j);
            }

            // If the current point is beyond the outer distance, break the inner loop
            if (r > outerDistance) {
                break;
            }
        }
    }

    // Resize the matrix to remove unused rows
    filteredPoints.conservativeResize(filteredRowCount, 3);

    return filteredPoints;
}

MatrixXi testSigmaPoints(const MatrixXf& selectedPoints, const MatrixXd& clusterBounds) {
    int numPoints = selectedPoints.rows();
    int numClusters = clusterBounds.rows();

    // Vector to store indices of filtered points
    vector<int> filteredIndices;

    for (int i = 0; i < numClusters; i++) {
        float azimMin = clusterBounds(i, 0);
        float azimMax = clusterBounds(i, 1);
        float elevMin = clusterBounds(i, 2);
        float elevMax = clusterBounds(i, 3);
        float innerDistance = clusterBounds(i, 4);
        float outerDistance = clusterBounds(i, 5);

        for (int j = 0; j < numPoints; j++) {
            float azim = selectedPoints(j, 1);
            float elev = selectedPoints(j, 2);
            float r = selectedPoints(j, 0);

            // Check if the point is within the cluster bounds
            if (azim >= azimMin && azim <= azimMax &&
                elev >= elevMin && elev <= elevMax &&
                r >= innerDistance && r <= outerDistance) {
                // Add the index to the filteredIndices vector
                filteredIndices.push_back(j);
            }

            // If the current point is beyond the outer distance, break the inner loop
            if (r > outerDistance) {
                break;
            }
        }
    }

    // Create a matrix from the indices
    MatrixXi filteredIndicesMatrix(filteredIndices.size(), 1);
    for (size_t i = 0; i < filteredIndices.size(); i++) {
        filteredIndicesMatrix(i, 0) = filteredIndices[i];
    }

    return filteredIndicesMatrix;
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    GLdouble cameraX = cameraDistance * sin(cameraAngleY * M_PI / 180.0) * cos(cameraAngleX * M_PI / 180.0);
    GLdouble cameraY = cameraDistance * cos(cameraAngleY * M_PI / 180.0) * cos(cameraAngleX * M_PI / 180.0);
    GLdouble cameraZ = cameraDistance * sin(cameraAngleX * M_PI / 180.0);

    gluLookAt(cameraX, cameraY, cameraZ, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    glEnable(GL_DEPTH_TEST);  // Enable depth testing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //draw points
    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_POINT_SMOOTH);
    drawPoints1();
    glDisable(GL_POINT_SMOOTH);

    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_POINT_SMOOTH);
    drawPoints2();
    glDisable(GL_POINT_SMOOTH);

    // //debug U, L ~~~~~~~~~~~~~~~~~~~
    // glColor3f(1.0, 1.0, 1.0);
    // glEnable(GL_POINT_SMOOTH);
    // drawTestPoints();
    // glDisable(GL_POINT_SMOOTH);
    // // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Draw single frustrum
    // glColor3f(0.8, 0.8, 1.0);
    // glEnable(GL_LINE_SMOOTH);
    // glLineWidth(2.0);
    // drawFrustum(azimuthalMin, azimuthalMax, 
    //             elevationMin, elevationMax,
    //             innerDistance, outerDistance);

    // Draw all voxels
    glColor3f(0.8, 0.8, 1.0);
    glEnable(GL_LINE_SMOOTH);
    // glLineWidth(2.0);
    for (size_t i = 0; i < clusterBounds.rows(); i++){
        //only draw if voxels are not flagged as empty
        if (clusterBounds(i,5) != 0){
            drawFrustum(clusterBounds(i,0), clusterBounds(i,1), clusterBounds(i,2), clusterBounds(i,3), clusterBounds(i,4), clusterBounds(i,5));
        }
    }

    // Draw ellipsoids
    array<float, 3> color1 = {0.1, 0.1, 1.0};
    for (size_t i = 0; i < ellipsoid1Means.size(); ++i) {
        drawEllipsoid(ellipsoid1Means[i], ellipsoid1Covariances[i], ellipsoid1Alphas[i], color1);
    }
    array<float, 3> color2 = {1., 0.1, 0.1};
    for (size_t i = 0; i < ellipsoid2Means.size(); ++i) {
        drawEllipsoid(ellipsoid2Means[i], ellipsoid2Covariances[i], ellipsoid2Alphas[i], color2);
    }

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL Error: " << gluErrorString(error) << std::endl;
    }

    glutSwapBuffers();
}

//function to load point cloud data from a csv
MatrixXf loadPointCloudCSV(string filename){
    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the CSV file." << std::endl;
    }
    // Parse and process the CSV file, skipping the first row
    csv::CSVReader reader(file, csv::CSVFormat().header_row(1).trim({}));
    
    // Initialize a vector to store rows temporarily
    std::vector<Eigen::Vector3f> rows;

    csv::CSVRow row;
    reader.read_row(row); // Skip the first row
    csv::CSVRow secondRow;
    reader.read_row(secondRow); // Skip the second row
    // Parse and process the CSV file
    // Iterate over rows and fill the vector
    for (csv::CSVRow& currentRow : reader) {
        // Assuming three columns in each row
        Eigen::Vector3f rowData;
        rowData << static_cast<float>(currentRow[8].get<int>()),
                   static_cast<float>(currentRow[9].get<int>()),
                   static_cast<float>(currentRow[10].get<int>());

        // Append the row to the vector
        rows.push_back(rowData);
    }

    // Close the file before processing the vector
    file.close();

    // Preallocate memory for the dataMatrix
    Eigen::MatrixXf dataMatrix(rows.size(), 3);

    // Copy the data from the vector to the dataMatrix
    for (size_t i = 0; i < rows.size(); ++i) {
        dataMatrix.row(i) = rows[i]/1000;
    }

    // points = dataMatrix;
    // return points;
    return dataMatrix;    // this seems to work?

    // // Extract every nth point using the colon operator
    // int n = 4;
    // Eigen::MatrixXf points = dataMatrix.block(0, 0, dataMatrix.rows() / n, dataMatrix.cols());
    // return points;
}

MatrixXf get_H(Eigen::Vector3f mu, Eigen::Vector3f angs){

    float phi = angs[0];
    float theta = angs[1];
    float psi = angs[2];    

    MatrixXf H(3,6);
    MatrixXf eye(3,3);
    eye << -1, 0, 0,
             0, -1, 0,
             0, 0, -1;
    H.block(0,0,3,3) << eye;

    // deriv of R() wrt phi.dot(mu)
    Eigen::MatrixXf Jx(3,3);
    Jx << 0., (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi)), (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi)),
          0., (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)), (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi)), 
          0., (-cos(phi)*cos(theta)), (-sin(phi)*cos(theta));
    Jx = Jx * mu;
    H.block(0, 3, 3, 1) = Jx;

    // deriv of R() wrt theta.dot(mu)
    Eigen::MatrixXf Jy(3,3);
    Jy << (-sin(theta)*cos(psi)), (cos(theta)*sin(phi)*cos(psi)), (-cos(theta)*cos(phi)*cos(psi)),
          (sin(psi)*sin(theta)), (-cos(theta)*sin(phi)*sin(psi)), (cos(theta)*sin(psi)*cos(phi)),
          (cos(theta)), (sin(phi)*sin(theta)), (-sin(theta)*cos(phi));
    Jy = Jy * mu;
    H.block(0, 4, 3, 1) = Jy;

    // deriv of R() wrt psi.dot(mu)
    Eigen::MatrixXf Jz(3,3);
    Jz << (-cos(theta)*sin(psi)), (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi)), (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi)),
         (-cos(psi)*cos(theta)), (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi)), (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi)),
         0., 0., 0.;
    Jz = Jz * mu;
    H.block(0, 5, 3, 1) = Jz;

    return H;
}

//given body frame xyz euler angles [phi, theta, psi], return 3x3 rotation matrix
Matrix3f R(float phi, float theta, float psi){
    MatrixXf mat(3,3); 
    mat << cos(theta)*cos(psi), sin(psi)*cos(phi)+sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi)-sin(theta)*cos(phi)*cos(psi),
           -sin(psi)*cos(theta), cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi)+sin(theta)*sin(psi)*cos(phi),
           sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta);

    return mat;
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("Points and Ellipsoids");
    glutReshapeWindow(1200, 800);
    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // Load Ouster Sample Dataset
    std::string csvFilePath1 = "sample_data/pcap_out_000106.csv";
    std::string csvFilePath2 = "sample_data/pcap_out_000107.csv";
    points = loadPointCloudCSV(csvFilePath1);
    points2 = loadPointCloudCSV(csvFilePath2);

    auto before = std::chrono::system_clock::now();
    auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);

    Eigen::MatrixXf pointsSpherical = cartesianToSpherical(points);
    // std::cout << "pointsSpherical: \n" << pointsSpherical.rows() << "\n";

    // Sort sphericalCoords based on radial distance
    vector<int> index(pointsSpherical.rows());
    iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), [&](int a, int b) {
        return pointsSpherical(a, 0) < pointsSpherical(b, 0); // Sort by radial distance
    });

    // Create a sorted matrix using the sorted indices
    MatrixXf sortedPointsSpherical(pointsSpherical.rows(), pointsSpherical.cols());
    for (int i = 0; i < pointsSpherical.rows(); i++) {
        sortedPointsSpherical.row(i) = pointsSpherical.row(index[i]);
    }

    // set up spherical voxel grid ~~~~~~~~~~~~~~~~~~~~~~~~~
    int numBinsPhi = 50;  // Adjust the number of bins as needed
    int numBinsTheta = 50; // Adjust the number of bins as needed
    int n = 40; // min size of the cluster
    float thresh = 0.3; // Threshold for radial distance
    float buff = 0.2; //buffer to add to inner and outer cluster range (helps attract nearby distributions)

    // init structure to store covariance data
    CovarianceMap sigma1;
    CovarianceMap sigma2;
    MeanMap mu1;
    MeanMap mu2;
    CovarianceMap L;
    CovarianceMap U;

    //count for figuring out how many voxles are occupied by scan1
    int occupiedCount = 0;

    //get spherical coordiantes and fit gaussians to points from first scan 
    vector<vector<vector<int>>> pointIndices = sortSphericalCoordinates(sortedPointsSpherical, numBinsTheta, numBinsPhi);
    for (int phi = 0; phi < numBinsPhi; phi++){
        for (int theta = 0; theta< numBinsTheta; theta++){
            // Retrieve the point indices inside angular bin
            const vector<int>& indices = pointIndices[theta][phi];

            // only calculate inner/outer bounds if there are a sufficient number of points in the spike 
            if (indices.size() > n) {
                // Use the indices to access the corresponding rows in sortedPointsSpherical
                MatrixXf selectedPoints = MatrixXf::Zero(indices.size(), sortedPointsSpherical.cols());
                for (int i = 0; i < indices.size(); ++i) {
                    selectedPoints.row(i) = sortedPointsSpherical.row(indices[i]);
                }

                // find inner and outer bounds for each theta/phi bin
                pair<float, float> clusterDistances = findCluster(selectedPoints, n, thresh, buff);
                innerDistance = clusterDistances.first;
                outerDistance = clusterDistances.second;

                //convert [desiredPhi][desiredTheta] to azimMin, azimMax, elevMin, elevMax
                float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
                float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
                float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
                float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;
                //hold on to these values
                clusterBounds.row(numBinsTheta*phi + theta) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;

                // find points from first scan inside voxel bounds and fit gaussians to each cluster
                MatrixXf filteredPoints = filterPointsInsideCluster(selectedPoints, clusterBounds.row(numBinsTheta*phi + theta));
                MatrixXf filteredPointsCart = sphericalToCartesian(filteredPoints);
                Eigen::VectorXf mean = filteredPointsCart.colwise().mean();
                Eigen::MatrixXf centered = filteredPointsCart.rowwise() - mean.transpose();
                Eigen::MatrixXf covariance = (centered.adjoint() * centered) / static_cast<float>(filteredPointsCart.rows() - 1);
                // std::cout << "Mean:\n" << mean << "\n\n";
                // std::cout << "Covariance:\n" << covariance << "\n";

                //hold on to means and covariances of clusters from scan1
                sigma1[theta][phi] = covariance;
                mu1[theta][phi] = mean;
                // cout << "\n mean: \n" << mu1[theta][phi] << "\n cov: \n" << sigma1[theta][phi] << "\n" << endl;


                // get U and L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
                Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
                Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors().real();
                U[theta][phi] = eigenvectors;
                // cout << "eigenvectors: " << U[theta][phi] << endl;
                // cout << "eigenval: " << eigenvalues[0] << endl;

                // create 6 2-sigma test points for each cluster and test to see if they fit inside the voxel
                MatrixXf axislen(3,3);
                axislen << eigenvalues[0], 0, 0,
                            0, eigenvalues[1], 0,
                            0, 0, eigenvalues[2];
                axislen = 2.0 * axislen.array().sqrt(); //theoretically should be *2 not *3 but this seems to work better

                MatrixXf rotated = axislen * U[theta][phi].transpose();

                Eigen::MatrixXf sigmaPoints(6,3);
                sigmaPoints.row(0) = mu1[theta][phi] + rotated.row(0).transpose(); //most compact axis
                sigmaPoints.row(1) = mu1[theta][phi] - rotated.row(0).transpose();
                sigmaPoints.row(2) = mu1[theta][phi] + rotated.row(1).transpose(); //middle
                sigmaPoints.row(3) = mu1[theta][phi] - rotated.row(1).transpose();
                sigmaPoints.row(4) = mu1[theta][phi] + rotated.row(2).transpose(); //largest axis
                sigmaPoints.row(5) = mu1[theta][phi] - rotated.row(2).transpose();

                // find out which test points fall inside the voxel bounds
                Eigen::MatrixXf sigmaPointsSpherical = cartesianToSpherical(sigmaPoints);
                MatrixXi sigmaPointsInside = testSigmaPoints(sigmaPointsSpherical, clusterBounds.row(numBinsTheta*phi + theta));
                // cout << "Sigma points inside: " << sigmaPointsInside.size() << "\n" << sigmaPointsInside << "\n" << endl;
                
                //see if each axis contains at least one test point within voxel
                if ((sigmaPointsInside.array() == 0).any() || (sigmaPointsInside.array() == 1).any()){
                    L[theta][phi].row(0) << 1, 0, 0; 
                } 
                else{
                    L[theta][phi].row(0) << 0, 0, 0;
                    testPoints.row(6*(numBinsTheta*phi + theta)) = sigmaPoints.row(0).transpose();
                    testPoints.row(6*(numBinsTheta*phi + theta)+1) = sigmaPoints.row(1).transpose();
                }
                if ((sigmaPointsInside.array() == 2).any() || (sigmaPointsInside.array() == 3).any()){
                    L[theta][phi].row(1) << 0, 1, 0; 
                } 
                else{
                    L[theta][phi].row(1) << 0, 0, 0;
                    testPoints.row(6*(numBinsTheta*phi + theta)+2) = sigmaPoints.row(2).transpose();
                    testPoints.row(6*(numBinsTheta*phi + theta)+3) = sigmaPoints.row(3).transpose();
                }
                if ((sigmaPointsInside.array() == 4).any() || (sigmaPointsInside.array() == 5).any()){
                    L[theta][phi].row(2) << 0, 0, 1; 
                } 
                else{
                    L[theta][phi].row(2) << 0, 0, 0;
                    testPoints.row(6*(numBinsTheta*phi + theta)+4) = sigmaPoints.row(4).transpose();
                    testPoints.row(6*(numBinsTheta*phi + theta)+5) = sigmaPoints.row(5).transpose();
                }

                // cout << "\n L: \n" << L[theta][phi] << endl;
                occupiedCount++; 
                // cout << "occupiedCount: " << occupiedCount << endl;
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                //update for drawing
                float alpha1 = 0.3f;
                ellipsoid1Means.push_back(mean);
                ellipsoid1Covariances.push_back(covariance);
                ellipsoid1Alphas.push_back(alpha1);

            }
            // use 0 value as a flag for unoccupied voxels
            else{
                innerDistance = 0;
                outerDistance = 0;
                float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
                float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
                float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
                float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;      
                clusterBounds.row(numBinsTheta*phi + theta) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;
            }

        }

    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    auto after1 = std::chrono::system_clock::now();
    auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
    auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
    cout << "Fit spherical voxels and guassians for scan 1 in: " << elapsedTimeMs << " ms" << endl;

    // Main Loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // apply transformation to points2
    
    // TODO: get actual rotation angles from last state estimate
    MatrixXf rot_mat = R(0.0f, 0.0f, 0.1f); 
    // cout << "rotation matrix: \n" << rot_mat << endl; 
    points2 << points2 * rot_mat;

    //fit points in scan2 to voxels
    Eigen::MatrixXf pointsSpherical2 = cartesianToSpherical(points2);
    // // Sort by radial distance -- not needed for scan2
    // vector<int> index2(pointsSpherical2.rows());
    // iota(index2.begin(), index2.end(), 0);
    // sort(index2.begin(), index2.end(), [&](int a, int b) {
    //     return pointsSpherical2(a, 0) < pointsSpherical2(b, 0); // Sort by radial distance
    // });
    // // Create a sorted matrix using the sorted indices
    // MatrixXf sortedPointsSpherical2(pointsSpherical2.rows(), pointsSpherical2.cols());
    // for (int i = 0; i < pointsSpherical2.rows(); i++) {
    //     sortedPointsSpherical2.row(i) = pointsSpherical2.row(index2[i]);
    // }

    // setup L, U matrices according to correspondenes in iteration i
    // init L_i and U_i to be bigger than they need to be (size of number of voxels occupied by scan 1 x3)
    Eigen::MatrixXf L_i(3*occupiedCount, 3);
    Eigen::MatrixXf U_i(3*occupiedCount, 3);
    
    // It is inefficient to construct the full (H^T W H) matrix direclty since W is very sparse
    // Instead we sum contributions from each voxel to a single 6x6 matrix to avoid memory inefficiency   
    Eigen::MatrixXf HTWH_i(6, 6);
    // cout << U_i.size() << endl;

    //fit gaussians
    int c = 0;
    // vector<vector<vector<int>>> pointIndices2 = sortSphericalCoordinates(sortedPointsSpherical2, numBinsTheta, numBinsPhi);
    vector<vector<vector<int>>> pointIndices2 = sortSphericalCoordinates(pointsSpherical2, numBinsTheta, numBinsPhi);
    for (int phi = 0; phi < numBinsPhi; phi++){
        for (int theta = 0; theta< numBinsTheta; theta++){
            // Retrieve the point indices inside angular bin
            const vector<int>& indices1 = pointIndices[theta][phi];
            const vector<int>& indices2 = pointIndices2[theta][phi];

            // only fit gaussians if there enough points from both scans 1 and 2 in the cell 
            if ((indices2.size() > n) && (indices1.size() > n)) {
                // Use the indices to access the corresponding rows in sortedPointsSpherical
                MatrixXf selectedPoints2 = MatrixXf::Zero(indices2.size(), pointsSpherical2.cols());
                for (int i = 0; i < indices2.size(); ++i) {
                    selectedPoints2.row(i) = pointsSpherical2.row(indices2[i]);
                }

                // find points from first scan inside voxel bounds and fit gaussians to each cluster
                MatrixXf filteredPoints2 = filterPointsInsideCluster(selectedPoints2, clusterBounds.row(numBinsTheta*phi + theta));
                MatrixXf filteredPointsCart2 = sphericalToCartesian(filteredPoints2);
                Eigen::VectorXf mean = filteredPointsCart2.colwise().mean();
                Eigen::MatrixXf centered = filteredPointsCart2.rowwise() - mean.transpose();
                Eigen::MatrixXf covariance = (centered.adjoint() * centered) / static_cast<float>(filteredPointsCart2.rows() - 1);

                //hold on to means and covariances of clusters from scan1
                sigma2[theta][phi] = covariance;
                mu2[theta][phi] = mean;
                
                //update L_i and U_i
                // cout << U[theta][phi] << endl;
                L_i.block(3*c, 0, 3, 3) << L[theta][phi];
                U_i.block(3*c, 0, 3, 3) << U[theta][phi];
                // cout << "U: " << typeid(U[theta][phi]).name() << endl;



                //add contributions to HTWH
                // Get noise components
                // TODO: the current weighting is slightly incorrect-- indices1.size() includes the number of all points in the radial bin (not just the ones within radial bounds)
                Eigen::MatrixXf R(3,3);
                // cout << filteredPoints2.size() << endl;
                // cout << indices2.size() << endl;
                R << (sigma1[theta][phi] / (indices1.size() - 1)) + (sigma2[theta][phi] / (indices2.size()-1));
                // cout << "R: \n" << R << endl;
                // use projection matrix to remove extended directions
                // R = L[theta][phi] * U[theta][phi].transpose() * R * U[theta][phi] * L[theta][phi].transpose();
                // Eigen::MatrixXf Ltest(2,3);
                // Ltest << 1, 0, 0,
                //          0, 1, 0;
                // R = Ltest * U[theta][phi].transpose() * R * U[theta][phi] * Ltest.transpose();
                // cout << "R: \n" << R << endl;
                // invert noise to get weighting
                Eigen::MatrixXf W = R.inverse();
                // cout << "W: \n" << W << endl;

                //get H matrix for voxel j
                Eigen::Vector3f angs = {0, 0, 1}; // TODO-- make rotation components of X
                Eigen::MatrixXf H_j = get_H(mu2[theta][phi], angs);
                // cout << H_j << endl;

                //suppress rows of H corresponding to overly extended directions
                Eigen::MatrixXf H_z = L[theta][phi] * U[theta][phi].transpose() * H_j;
                // cout << "\n L: \n" << L[theta][phi] << "\n H: " << endl;
                // cout << H_z << endl;

                //put together HTWH for voxel j and contribute to total HTWH_i (for all voxels of current iteration)
                Eigen::MatrixXf HTWH_j = H_z.transpose() * W * H_z;
                HTWH_i = HTWH_i + HTWH_j;



                c++;

                //update for drawing
                float alpha2 = 0.3f;
                ellipsoid2Means.push_back(mean);
                ellipsoid2Covariances.push_back(covariance);
                ellipsoid2Alphas.push_back(alpha2);
            }
        }
    }

    // perform "conservative resize" on L_i and U_i before doing any calculations
    // cout << "\n L_i before: \n " << L_i << endl;
    L_i.conservativeResize(3*c, Eigen::NoChange);
    U_i.conservativeResize(3*c, Eigen::NoChange);
    // cout << "\n L_i after: \n " << L_i << endl;

    // (H_z^T*W*H_z)

    // GOAL: find linear perterbation dX to correct previous state estimate X
    // dx = (H^T * W * H)^-1 * H^T * W * deltaY
    // H -> appended jacobain matrix (3*occupiedCount, 6)
    // W -> block diagonal weighting matrix (3*occupiedCount, 3*occupiedCount)


    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    auto after2 = std::chrono::system_clock::now();
    auto after2Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after2);
    auto elapsedTimeMs2 = std::chrono::duration_cast<std::chrono::milliseconds>(after2Ms - after1Ms).count();
    cout << "Fit spherical voxels and guassians for scan 2 in: " << elapsedTimeMs2 << " ms" << endl;

    glEnable(GL_DEPTH_TEST);

    // Initialize VBOs and VAOs
    // initVBOsAndVAOs();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glutMainLoop();

    // Clean up VBOs and VAOs
    glDeleteBuffers(1, &pointsVBO);
    // glDeleteBuffers(1, &ellipsoidVBO);
    glDeleteBuffers(1, &frustumVBO);
    glDeleteVertexArrays(1, &pointsVAO);
    // glDeleteVertexArrays(1, &ellipsoidVAO);
    glDeleteVertexArrays(1, &frustumVAO);


    return 0;
}
