#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>

GLuint pointsVBO, ellipsoidVBO, pointsVAO, ellipsoidVAO;

GLdouble cameraDistance = 10.0;
GLdouble cameraAngleX = 45.0;
GLdouble cameraAngleY = -45.0;

GLdouble lastMouseX = 0.0;
GLdouble lastMouseY = 0.0;

Eigen::MatrixXf points(10, 3);  // Declare points as a global variable

std::vector<Eigen::Vector3f> ellipsoidMeans;
std::vector<Eigen::Matrix3f> ellipsoidCovariances;
std::vector<float> ellipsoidAlphas;

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

void initVBOsAndVAOs() {
    // // Create VBO and VAO for points
    createVBO(pointsVBO, GL_ARRAY_BUFFER, points);
    createVAO(pointsVAO, pointsVBO, 3, 0, points.rows());  // Pass the size of points matrix
    // Create VBO and VAO for points
    // createVBO(pointsVBO, GL_ARRAY_BUFFER, points);
    // createVAO(pointsVAO, pointsVBO, 3, 0);

    // Create VBO and VAO for ellipsoids
    Eigen::MatrixXf ellipsoidVertices(100, 3);  // Adjust the number of vertices as needed
    for (int i = 0; i < 100; ++i) {
        float theta = 2.0 * M_PI * i / 100;
        ellipsoidVertices(i, 0) = cos(theta);
        ellipsoidVertices(i, 1) = sin(theta);
        ellipsoidVertices(i, 2) = 0.0f;
    }

    createVBO(ellipsoidVBO, GL_ARRAY_BUFFER, ellipsoidVertices);
    createVAO(ellipsoidVAO, ellipsoidVBO, 3, 1, ellipsoidVertices.rows());  // Pass the size of ellipsoidVertices matrix
}


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

void drawEllipsoid(const Eigen::Vector3f& mean, const Eigen::Matrix3f& covariance, GLfloat alpha) {
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

    glColor4f(0.0, 0.0, 1.0, alpha);
    glutSolidSphere(1.0, 100, 100);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glPopMatrix();
}

// void drawPoints() {
//     std::cout << "Drawing Points. Size: " << points.rows() << std::endl;
//     glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
//     glBindVertexArray(pointsVAO);
//     glPointSize(5.0f);  // Increase point size
//     glColor3f(1.0, 1.0, 1.0);
//     // glColor3f(1.0, 0.0, 0.0);  // Red points


//     glDrawArrays(GL_POINTS, 0, points.rows());
//     glBindVertexArray(0);
//     glBindBuffer(GL_ARRAY_BUFFER, 0);
// }

void drawPoints() {
    std::cout << "Drawing Points. Size: " << points.rows() << std::endl;
    glPointSize(5.0f);
    glColor3f(1.0, 1.0, 1.0);  // White color

    glBegin(GL_POINTS);
    for (int i = 0; i < points.rows(); ++i) {
        glVertex3f(points(i, 0), points(i, 1), points(i, 2));
    }
    glEnd();
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    GLdouble cameraX = cameraDistance * sin(cameraAngleY * M_PI / 180.0) * cos(cameraAngleX * M_PI / 180.0);
    GLdouble cameraY = cameraDistance * sin(cameraAngleX * M_PI / 180.0);
    GLdouble cameraZ = cameraDistance * cos(cameraAngleY * M_PI / 180.0) * cos(cameraAngleX * M_PI / 180.0);

    gluLookAt(cameraX, cameraY, cameraZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    glEnable(GL_DEPTH_TEST);  // Enable depth testing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //draw points
    // glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_POINT_SMOOTH);
    drawPoints();
    glDisable(GL_POINT_SMOOTH);

    // Draw ellipsoids
    for (size_t i = 0; i < ellipsoidMeans.size(); ++i) {
        drawEllipsoid(ellipsoidMeans[i], ellipsoidCovariances[i], ellipsoidAlphas[i]);
    }

    glDisable(GL_BLEND);
    // glDisable(GL_DEPTH_TEST);  // Disable depth testing after drawing

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL Error: " << gluErrorString(error) << std::endl;
    }

    glutSwapBuffers();
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
    cameraAngleY -= (x - lastMouseX) * 0.1;
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

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("Points and Ellipsoids");
    glutReshapeWindow(800, 600);

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // Initialize VBOs and VAOs
    initVBOsAndVAOs();

    // Initialize points
    Eigen::Matrix3f customCovariance;
    customCovariance << 1.0f, 0.5f, 0.3f,
                        0.5f, 2.0f, 0.8f,
                        0.3f, 0.8f, 1.5f;
    Eigen::Vector3f customMean(1.0f, 0.0f, 0.0f);
    points = generateEigenCovariance(10000, customMean, customCovariance);
    std::cout << "Random Points:\n" << points << "\n";

    // Estimate mean and covariance of random points
    Eigen::VectorXf mean = points.colwise().mean();
    Eigen::MatrixXf centered = points.rowwise() - mean.transpose();
    Eigen::MatrixXf covariance = (centered.adjoint() * centered) / static_cast<float>(points.rows() - 1);
    std::cout << "Mean:\n" << mean << "\n\n";
    std::cout << "Covariance:\n" << covariance << "\n";

    // Initialize ellipsoids with specified values ~~~~~~~~~~~
    Eigen::Vector3f mean1 = mean;
    Eigen::MatrixXf covariance1 = covariance;
    float alpha1 = 0.5f;
    ellipsoidMeans.push_back(mean1);
    ellipsoidCovariances.push_back(covariance1);
    ellipsoidAlphas.push_back(alpha1);

    // Eigen::Vector3f mean2(-1.0f, 0.0f, 0.0f);
    // Eigen::Matrix3f covariance2;
    // covariance2 << 2.0f, 0.3f, 0.5f,
    //                0.3f, 1.5f, 0.8f,
    //                0.5f, 0.8f, 2.0f;
    // float alpha2 = 0.7f;
    // ellipsoidMeans.push_back(mean2);
    // ellipsoidCovariances.push_back(covariance2);
    // ellipsoidAlphas.push_back(alpha2);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 0; i <= 100; i++){
        //generate random covariance
        Eigen::Matrix3f randomCovariance;
        // randomCovariance.setRandom();
        randomCovariance << 0.3f, 0.f, 0.f,
                   0.f, 0.3f, 0.f,
                   0.f, 0.f, 0.3f;
        //generate random mean
        Eigen::Vector3f randomMean;
        randomMean.setRandom();
        randomMean *= 30;
        float randomAlpha = 0.5; 

        // std::cout << "randomMean:\n" << randomMean << "\n\n";
        std::cout << "randomCovariance:\n" << randomCovariance << "\n\n";

        //push back to global vars
        ellipsoidMeans.push_back(randomMean);
        ellipsoidCovariances.push_back(randomCovariance);
        ellipsoidAlphas.push_back(randomAlpha);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    glDeleteBuffers(1, &ellipsoidVBO);
    glDeleteVertexArrays(1, &pointsVAO);
    glDeleteVertexArrays(1, &ellipsoidVAO);


    return 0;
}
