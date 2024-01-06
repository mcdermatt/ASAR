#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>

GLdouble cameraDistance = 10.0;
GLdouble cameraAngleX = 45.0;
GLdouble cameraAngleY = -45.0;

GLdouble lastMouseX = 0.0;
GLdouble lastMouseY = 0.0;

Eigen::MatrixXf points(10, 3);  // Declare points as a global variable

std::vector<Eigen::Vector3f> ellipsoidMeans;
std::vector<Eigen::Matrix3f> ellipsoidCovariances;
std::vector<float> ellipsoidAlphas;

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

    glPopMatrix();
}

void drawPoints(const Eigen::MatrixXf& eigenMatrix) {
    glPointSize(1.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0, 1.0, 1.0);  // Set color to white for points
    for (int i = 0; i < eigenMatrix.rows(); ++i) {
        glVertex3f(eigenMatrix(i, 0), eigenMatrix(i, 1), eigenMatrix(i, 2));
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
    glColor3f(1.0, 1.0, 1.0);
    drawPoints(points);

    // Draw ellipsoids
    for (size_t i = 0; i < ellipsoidMeans.size(); ++i) {
        drawEllipsoid(ellipsoidMeans[i], ellipsoidCovariances[i], ellipsoidAlphas[i]);
    }

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);  // Disable depth testing after drawing

    glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<GLdouble>(w) / h, 1.0, 100.0);
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
    glEnable(GL_DEPTH_TEST);

    // Initialize points
    Eigen::Matrix3f customCovariance;
    customCovariance << 1.0f, 0.5f, 0.3f,
                        0.5f, 2.0f, 0.8f,
                        0.3f, 0.8f, 1.5f;
    Eigen::Vector3f customMean(1.0f, 0.0f, 0.0f);
    points = generateEigenCovariance(100000, customMean, customCovariance);
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
    for (int i = 0; i <= 10; i++){
        //generate random covariance
        Eigen::Matrix3f randomCovariance;
        // randomCovariance.setRandom();
        randomCovariance << 0.3f, 0.f, 0.f,
                   0.f, 0.3f, 0.f,
                   0.f, 0.f, 0.3f;
        //generate random mean
        Eigen::Vector3f randomMean;
        randomMean.setRandom();
        randomMean *= 5;
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

    return 0;
}
