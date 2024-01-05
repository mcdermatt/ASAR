#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>

//attempt using glfw (more control but more buggy)

GLFWwindow* window;
GLuint shaderProgram;  // Declare shader program variable

// Declare matrices at a higher scope
glm::mat4 projectionMatrix;
glm::mat4 viewMatrix;

GLuint vboPoints; // Vertex Buffer Object for points
GLuint vaoPoints; // Vertex Array Object for points

GLuint vboEllipsoid; // Vertex Buffer Object for ellipsoid
GLuint vaoEllipsoid; // Vertex Array Object for ellipsoid

// Camera parameters
glm::vec3 cameraPosition = glm::vec3(3.0f, 3.0f, 10.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

GLfloat yaw = -90.0f;
GLfloat pitch = 0.0f;
GLfloat lastX = 400.0f;
GLfloat lastY = 300.0f;
GLboolean firstMouse = true;

GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;
GLfloat cameraSpeed = 5.0f * 0.01f;

// Declare vectors at a higher scope
std::vector<GLfloat> verticesEllipsoid;
// Declare Eigen matrix at a higher scope
Eigen::MatrixXf eigenMatrix;


void errorCallback(int error, const char* description) {
    std::cerr << "Error: " << description << std::endl;
}

void mouse_callback(GLFWwindow* window, double xPos, double yPos) {
    if (firstMouse) {
        lastX = xPos;
        lastY = yPos;
        firstMouse = false;
    }

    GLfloat xOffset = xPos - lastX;
    GLfloat yOffset = lastY - yPos;

    lastX = xPos;
    lastY = yPos;

    const GLfloat sensitivity = 0.10f;
    xOffset *= sensitivity;
    yOffset *= sensitivity;

    yaw += xOffset;
    pitch += yOffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);

    // Update view matrix
    viewMatrix = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);

    std::cout << "Yaw: " << yaw << ", Pitch: " << pitch << std::endl;
}


void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPosition += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPosition -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPosition -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPosition += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;\
    viewMatrix = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}


GLuint compileShader(GLenum shaderType, const char* shaderSource, const char* shaderName) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, nullptr);
    glCompileShader(shader);

    // Check shader compilation status
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed (" << shaderName << "): " << infoLog << std::endl;
        return 0;
    }

    return shader;
}

GLuint linkShaderProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check shader program linking status
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return 0;
    }

    return program;
}

void setupBuffersEllipsoid() {
    // Generate ellipsoid vertices (adjust the parameters as needed)
    const int stacks = 30;
    const int slices = 30;
    const float radiusX = 1.0f;
    const float radiusY = 2.0f;
    const float radiusZ = 3.0f;

    for (int i = 0; i <= stacks; ++i) {
        double phi = i * M_PI / stacks;
        double sinPhi = sin(phi);
        double cosPhi = cos(phi);

        for (int j = 0; j <= slices; ++j) {
            double theta = j * 2 * M_PI / slices;
            double sinTheta = sin(theta);
            double cosTheta = cos(theta);

            double x = radiusX * cosTheta * sinPhi;
            double y = radiusY * cosPhi;
            double z = radiusZ * sinTheta * sinPhi;

            verticesEllipsoid.push_back(x);
            verticesEllipsoid.push_back(y);
            verticesEllipsoid.push_back(z);
        }
    }

    // Generate Vertex Buffer Object (VBO) for ellipsoid
    glGenBuffers(1, &vboEllipsoid);
    glBindBuffer(GL_ARRAY_BUFFER, vboEllipsoid);
    glBufferData(GL_ARRAY_BUFFER, verticesEllipsoid.size() * sizeof(GLfloat), verticesEllipsoid.data(), GL_STATIC_DRAW);

    // Generate Vertex Array Object (VAO) for ellipsoid
    glGenVertexArrays(1, &vaoEllipsoid);
    glBindVertexArray(vaoEllipsoid);

    // Specify the layout of the vertex data for ellipsoid
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Unbind VAO and VBO for ellipsoid
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void renderEllipsoid() {
    // Bind VAO for ellipsoid
    glBindVertexArray(vaoEllipsoid);

    // Render ellipsoid
    // glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vaoEllipsoid)); //old
    // glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vaoEllipsoid)); //new
    // Render ellipsoid
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verticesEllipsoid.size() / 3));

    // Unbind VAO for ellipsoid
    glBindVertexArray(0);
}

void setupBuffers() {
    // Create an n by 3 matrix of random floats
    eigenMatrix = Eigen::MatrixXf::Random(77777, 3);

    // Convert Eigen matrix to a std::vector
    std::vector<GLfloat> points(eigenMatrix.data(), eigenMatrix.data() + eigenMatrix.size());

    // Generate Vertex Buffer Object (VBO) for points
    glGenBuffers(1, &vboPoints);
    glBindBuffer(GL_ARRAY_BUFFER, vboPoints);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), points.data(), GL_STATIC_DRAW);

    // Generate Vertex Array Object (VAO) for points
    glGenVertexArrays(1, &vaoPoints);
    glBindVertexArray(vaoPoints);

    // Specify the layout of the vertex data for points
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Unbind VAO and VBO for points
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void renderPoints() {
    // Bind VAO for points
    glBindVertexArray(vaoPoints);

    // Render points
    // glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vaoPoints));
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(eigenMatrix.size() / 3));

    // Unbind VAO for points
    glBindVertexArray(0);
}

int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW initialization failed" << std::endl;
        return -1;
    }

    glfwSetErrorCallback(errorCallback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    window = glfwCreateWindow(800, 600, "OpenGL Points", NULL, NULL);
    if (!window) {
        std::cerr << "Window creation failed" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE; // Enable experimental features for GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW initialization failed" << std::endl;
        return -1;
    }
    glfwSetKeyCallback(window, keyCallback);
    // Set mouse callback
    glfwSetCursorPosCallback(window, mouse_callback);

    glEnable(GL_DEPTH_TEST);

    // Vertex shader source code
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 projectionMatrix;
        uniform mat4 viewMatrix;
        void main() {
            gl_Position = projectionMatrix * viewMatrix * vec4(aPos, 1.0);
        }
    )";

    // Fragment shader source code
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(gl_FragCoord.x / 800.0, gl_FragCoord.y / 600.0, 0.0, 1.0);
        }
    )";

    // Compile shaders
    // GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    // GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource, "Vertex Shader");
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource, "Fragment Shader");

    if (vertexShader == 0 || fragmentShader == 0) {
        // Shader compilation failed
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Link shader program
    shaderProgram = linkShaderProgram(vertexShader, fragmentShader);

    if (shaderProgram == 0) {
        // Shader program linking failed
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Matrix setup: Define a basic perspective projection matrix
    float fov = glm::radians(45.0f);  // Field of view in degrees
    float aspectRatio = 800.0f / 600.0f;  // Width divided by height
    float nearClip = 0.1f;
    float farClip = 1000.0f;

    projectionMatrix = glm::perspective(fov, aspectRatio, nearClip, farClip);
    // viewMatrix = glm::lookAt(glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    viewMatrix = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    cameraFront = glm::normalize(cameraFront);  // Normalize cameraFront vector

    // Setup VBO and VAO for points
    setupBuffers();

    // Setup VBO and VAO for ellipsoid
    setupBuffersEllipsoid();


    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // White background
        glViewport(0, 0, 800, 600); // Adjust the values based on your window size

        // Handle input
        processInput(window);

        // Update view matrix based on camera parameters
        viewMatrix = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);
        // std::cout << "View Matrix:\n";
        // for (int i = 0; i < 4; ++i) {
        //     for (int j = 0; j < 4; ++j) {
        //         std::cout << viewMatrix[j][i] << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << std::endl;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);

        std::cout << "Camera Front: (" << cameraFront.x << ", " << cameraFront.y << ", " << cameraFront.z << ")" << std::endl;

        // Set the uniform in the shader with the updated viewMatrix
        glUseProgram(shaderProgram);
        GLint viewMatrixLoc = glGetUniformLocation(shaderProgram, "viewMatrix");
        glUniformMatrix4fv(viewMatrixLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
        std::cout << "View Matrix:\n";
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << viewMatrix[j][i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;

        // Set the uniform in the shader with the projectionMatrix
        GLint projectionMatrixLoc = glGetUniformLocation(shaderProgram, "projectionMatrix");
        glUniformMatrix4fv(projectionMatrixLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

        glUseProgram(0);

        // Call the rendering function for ellipsoid
        renderEllipsoid();

        // Call the rendering function for points
        renderPoints();

        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cerr << "OpenGL error after rendering: " << error << std::endl;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &vaoPoints);
    glDeleteBuffers(1, &vboPoints);
    glDeleteVertexArrays(1, &vaoEllipsoid);
    glDeleteBuffers(1, &vboEllipsoid);
    glDeleteProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
