#include "visualization.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Static pointer to hold the instance of the visualization object
visualization* visualizationInstance = nullptr;

visualization::visualization() { //int argc, char** argv
    // Constructor implementation
    cameraDistance = 10.0; // Example initial value
    cameraAngleX = 45.0;   // Example initial value
    cameraAngleY = 45.0;   // Example initial value
    lastMouseX = 0;
    lastMouseY = 0;
    initializeOpenGL(); //argc, argv
    visualizationInstance = this; // Set the instance pointer
}

visualization::~visualization() {
    // Destructor implementation
}

void visualization::initializeOpenGL() { //int argc, char** argv
    // Implement OpenGL initialization

    //won't have access to command line arguments anyways
    int argc = 1;
    char *argv[1] = {(char *)"program"};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ICET Visualizer");
    glutReshapeWindow(1200, 800);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
    }

    std::cout << "OpenGL initialized." << std::endl;
}

void visualization::display(){
    if (visualizationInstance){
        visualizationInstance->render();
    }
}


void visualization::reshape(int w, int h) {
    if (visualizationInstance) {
        visualizationInstance->handleReshape(w, h);
    }
}

void visualization::specialKeys(int key, int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleSpecialKeys(key, x, y);
    }
}

void visualization::mouse(int button, int state, int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleMouse(button, state, x, y);
    }
}

void visualization::motion(int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleMotion(x, y);
    }
}

void visualization::keyboard(unsigned char key, int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleKeyboard(key, x, y);
    }
}

void visualization::handleReshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<GLdouble>(w) / h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
}

void visualization::handleSpecialKeys(int key, int x, int y) {
    if (key == GLUT_KEY_UP) {
        cameraDistance -= 0.5;
    } else if (key == GLUT_KEY_DOWN) {
        cameraDistance += 0.5;
    }
    glutPostRedisplay();
}

void visualization::handleMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
    }
}

void visualization::handleMotion(int x, int y) {
    cameraAngleY += (x - lastMouseX) * 0.1;
    cameraAngleX += (y - lastMouseY) * 0.1;

    lastMouseX = x;
    lastMouseY = y;

    glutPostRedisplay();
}

void visualization::handleKeyboard(unsigned char key, int x, int y) {
    if (key == 27) {
        exit(0);
    }
}

void visualization::render() {
    // Implement rendering using OpenGL
    // std::cout << "Rendering visualization." << std::endl;

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

// Dummy implementations for drawing functions
void visualization::drawPoints1() {
    glPointSize(3.0f);
    glColor3f(0.5, 0.5, 1.0); // Blue color

    glBegin(GL_POINTS);
    for (int i = 0; i < points1.rows(); ++i) {
        glVertex3f(points1(i, 0), points1(i, 1), points1(i, 2));
    }
    glEnd();
}

void visualization::drawPoints2() {
    // Implement drawing of points
}

void visualization::drawFrustum(float x, float y, float z, float w, float h, float d) {
    // Implement drawing of frustum
}

void visualization::drawEllipsoid(const std::array<float, 3>& mean, const std::array<float, 9>& covariance, float alpha, const std::array<float, 3>& color) {
    // Implement drawing of ellipsoid
}