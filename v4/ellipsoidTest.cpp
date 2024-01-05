#include <GL/glut.h>
#include <cmath>
#include <cstdio> 

const double PI = 3.141592653589793;

double cameraDistance = 5.0;
double cameraAngleX = 45.0;
double cameraAngleY = 45.0;

void drawEllipsoid(double a, double b, double c, int stacks, int slices) {
    glPushMatrix();
    glTranslatef(0.0, 0.0, 0.0);
    glScalef(a, b, c);

    for (int i = 0; i <= stacks; ++i) {
        double phi = i * PI / stacks;
        double sinPhi = sin(phi);
        double cosPhi = cos(phi);

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; ++j) {
            double theta = j * 2 * PI / slices;
            double sinTheta = sin(theta);
            double cosTheta = cos(theta);

            double x = cosTheta * sinPhi;
            double y = cosPhi;
            double z = sinTheta * sinPhi;

            glNormal3f(x, y, z);
            glVertex3f(x, y, z);

            x = cosTheta * sin(phi + PI / stacks);
            y = cos(phi + PI / stacks);
            z = sinTheta * sin(phi + PI / stacks);

            glNormal3f(x, y, z);
            glVertex3f(x, y, z);
        }
        glEnd();
    }

    glPopMatrix();

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL Error in drawEllipsoid: %s\n", gluErrorString(error));
    }
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    double cameraX = cameraDistance * sin(cameraAngleY * PI / 180.0) * cos(cameraAngleX * PI / 180.0);
    double cameraY = cameraDistance * sin(cameraAngleX * PI / 180.0);
    double cameraZ = cameraDistance * cos(cameraAngleY * PI / 180.0) * cos(cameraAngleX * PI / 180.0);

    gluLookAt(cameraX, cameraY, cameraZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // flat lighting
    glShadeModel(GL_FLAT);
    //shaded
    // glEnable(GL_LIGHTING);
    // glEnable(GL_LIGHT0);
    // GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
    // glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    // glColor3f(0.7, 0.7, 1.0);     // just rgb

    // Set RGBA color with alpha (transparency)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.7, 0.7, 1.0, 0.5); // Adjust the alpha value (0.0 to 1.0)
    drawEllipsoid(1.0, 2.0, 3.0, 50, 50);

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL Error: %s\n", gluErrorString(error));
    }

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

void mouse(int button, int state, int x, int y) {
    // You can handle mouse events here, if needed
}

void motion(int x, int y) {
    // You can handle mouse motion events here, if needed
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow("Ellipsoid Plot");
    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutMainLoop();

    return 0;
}
