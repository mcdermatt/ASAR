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
    glPointSize(3.0f);
    glColor3f(1., 0.5, 0.5);  // red
    glBegin(GL_POINTS);
    for (int i = 0; i < points2.rows(); ++i) {
        glVertex3f(points2(i, 0), points2(i, 1), points2(i, 2));
    }
    glEnd();
}

void visualization::drawCroppedSphere(GLfloat radius, GLfloat azimuthalMin, GLfloat azimuthalMax,
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
}

void visualization::drawFrustum(GLfloat azimuthalMin, GLfloat azimuthalMax, 
                                GLfloat elevationMin, GLfloat elevationMax,
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

void visualization::drawEllipsoid(const Eigen::Vector3f& mean, const Eigen::Matrix3f& covariance, GLfloat alpha, const std::array<float, 3>& color) {
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
    glutSolidSphere(1.0, 30, 30); //slow high res
    // glutSolidSphere(1.0, 10, 10); //faster low res

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glPopMatrix();
}