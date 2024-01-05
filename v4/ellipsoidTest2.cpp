#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>

GLdouble cameraDistance = 10.0;
GLdouble cameraAngleX = 45.0;
GLdouble cameraAngleY = -45.0;

GLdouble lastMouseX = 0.0;
GLdouble lastMouseY = 0.0;

Eigen::MatrixXf points(10, 3);  // Declare points as a global variable

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

void drawEllipsoid(GLdouble a, GLdouble b, GLdouble c, int stacks, int slices, GLfloat alpha) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPushMatrix();
    glScalef(a, b, c);

    for (int i = 0; i <= stacks; ++i) {
        double phi = i * M_PI / stacks;
        double sinPhi = sin(phi);
        double cosPhi = cos(phi);

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; ++j) {
            double theta = j * 2 * M_PI / slices;
            double sinTheta = sin(theta);
            double cosTheta = cos(theta);

            double x = cosTheta * sinPhi;
            double y = cosPhi;
            double z = sinTheta * sinPhi;

            glNormal3f(x, y, z);
            glColor4f(0.0, 0.0, 1.0, alpha);
            glVertex3f(x, y, z);

            x = cosTheta * sin(phi + M_PI / stacks);
            y = cos(phi + M_PI / stacks);
            z = sinTheta * sin(phi + M_PI / stacks);

            glNormal3f(x, y, z);
            glColor4f(0.0, 0.0, 1.0, alpha);
            glVertex3f(x, y, z);
        }
        glEnd();
    }

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
    glColor4f(0.0, 0.0, 1.0, 0.5);
    drawEllipsoid(1.0, 2.0, 3.0, 50, 50, 0.5);

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

    //uniform cube
    // points = Eigen::MatrixXf::Random(100000, 3);
    //gaussian
    points = generateEigenNormal(100000, 3, 0.0f, 1.0f);

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
