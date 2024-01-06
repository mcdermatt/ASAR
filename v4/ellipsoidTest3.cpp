#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>

GLdouble cameraDistance = 10.0;
GLdouble cameraAngleX = 45.0;
GLdouble cameraAngleY = -45.0;

GLdouble lastMouseX = 0.0;
GLdouble lastMouseY = 0.0;

Eigen::MatrixXf points(10, 3);  // Declare points as a global variable

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

void setupEllipsoid(Eigen::Vector3f& mean, Eigen::Matrix3f& covariance) {
    // Example: Set mean and covariance
    mean << 1.0f, 0.0f, 0.0f;
    covariance << 1.0f, 0.5f, 0.3f,
                   0.5f, 2.0f, 0.8f,
                   0.3f, 0.8f, 1.5f;
}

#include <GL/glut.h>
#include <Eigen/Dense>

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

    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0, 1.0, 1.0);
    drawPoints(points);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat customAlpha = 0.5;
    Eigen::Vector3f customMean;
    Eigen::Matrix3f customCovariance;
    setupEllipsoid(customMean, customCovariance);
    drawEllipsoid(customMean, customCovariance, customAlpha);
    drawEllipsoid(Eigen::Vector3f (1.0f, 2.0f, 0.0f), customCovariance, customAlpha); // draw 2nd random ellipsoid

    glutSwapBuffers();
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<double>(width) / static_cast<double>(height), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

void specialKeys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP:
        cameraAngleX += 5.0;
        break;
    case GLUT_KEY_DOWN:
        cameraAngleX -= 5.0;
        break;
    case GLUT_KEY_LEFT:
        cameraAngleY -= 5.0;
        break;
    case GLUT_KEY_RIGHT:
        cameraAngleY += 5.0;
        break;
    }

    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 'w':
        cameraDistance -= 0.5;
        break;
    case 's':
        cameraDistance += 0.5;
        break;
    }

    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
    }
}

void motion(int x, int y) {
    GLdouble deltaX = x - lastMouseX;
    GLdouble deltaY = y - lastMouseY;

    lastMouseX = x;
    lastMouseY = y;

    cameraAngleY -= deltaX * 0.2;
    cameraAngleX += deltaY * 0.2;

    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("Points and Ellipsoid");
    glutReshapeWindow(800, 600);
    glEnable(GL_DEPTH_TEST);

    // Generate random points
    Eigen::Matrix3f customCovariance;
    customCovariance << 1.0f, 0.5f, 0.3f,
                    0.5f, 2.0f, 0.8f,
                    0.3f, 0.8f, 1.5f;
    Eigen::Vector3f customMean(1.0f, 0.0f, 0.0f);
    points = generateEigenCovariance(100000, customMean, customCovariance);
    // std::cout << "Random Points:\n" << points << "\n";
    std::cout << "customMean:\n" << customMean << "\n";
    std::cout << "customCovariance:\n" << customCovariance << "\n";


    // Setup ellipsoid parameters
    Eigen::Vector3f ellipsoidMean;
    Eigen::Matrix3f ellipsoidCovariance;
    setupEllipsoid(ellipsoidMean, ellipsoidCovariance);

    std::cout << "Ellipsoid Mean:\n" << ellipsoidMean << "\n";
    std::cout << "Ellipsoid Covariance:\n" << ellipsoidCovariance << "\n";

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
