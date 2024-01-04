#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <Eigen/Dense>

// g++ -o oglTest oglTest.cpp -lglfw -lGLEW -lGL -lGLU -L/usr/include/glm

GLFWwindow* window;
GLuint shaderProgram;  // Declare shader program variable

// Declare matrices at a higher scope
glm::mat4 projectionMatrix;
glm::mat4 viewMatrix;
glm::mat4 modelMatrix;

GLuint vbo; // Vertex Buffer Object
GLuint vao; // Vertex Array Object
GLuint vboEllipsoid; // Vertex Buffer Object for ellipsoids
GLuint vaoEllipsoid; // Vertex Array Object for ellipsoids

std::vector<GLfloat> ellipsoidPoints;

// Camera parameters
glm::vec3 cameraPosition = glm::vec3(3.0f, 3.0f, 3.0f);
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

struct Ellipsoid {
    glm::vec3 position;
    glm::vec3 color;
};

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

    const GLfloat sensitivity = 0.05f;
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
        cameraPosition += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
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

    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        glDeleteProgram(program);  // Delete the program to avoid a resource leak
        return 0;
    }



    return program;
}

void setupBuffers() {
    // Create an n by 3 matrix of random floats for points
    Eigen::MatrixXf pointsMatrix = Eigen::MatrixXf::Random(77777, 3);
    std::vector<GLfloat> points(pointsMatrix.data(), pointsMatrix.data() + pointsMatrix.size());

    // Generate Vertex Buffer Object (VBO)
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), points.data(), GL_STATIC_DRAW);

    // Generate Vertex Array Object (VAO)
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Specify the layout of the combined vertex data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Unbind VAO and VBO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void setupEllipsoidBuffer() {

    // Create ellipsoid data with position and color
    std::vector<Ellipsoid> ellipsoidData;
    for (int i = 0; i < 5; ++i) {
        Ellipsoid ellipsoid;
        ellipsoid.position = glm::vec3(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f,
                                       2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f,
                                       2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
        // ellipsoid.color = glm::vec3(static_cast<float>(rand()) / RAND_MAX,
        //                             static_cast<float>(rand()) / RAND_MAX,
        //                             static_cast<float>(rand()) / RAND_MAX);
        ellipsoid.color = glm::vec3(static_cast<float>(0.5) / 1,
                            static_cast<float>(0.5) / 1,
                            static_cast<float>(0.5) / 1);
        ellipsoidData.push_back(ellipsoid);
    }

    // Extract position and color vectors from ellipsoid data
    std::vector<GLfloat> ellipsoidPoints;
    for (const auto& ellipsoid : ellipsoidData) {
        ellipsoidPoints.push_back(ellipsoid.position.x);
        ellipsoidPoints.push_back(ellipsoid.position.y);
        ellipsoidPoints.push_back(ellipsoid.position.z);
        ellipsoidPoints.push_back(ellipsoid.color.r);
        ellipsoidPoints.push_back(ellipsoid.color.g);
        ellipsoidPoints.push_back(ellipsoid.color.b);
    }

    // Print ellipsoid data for debugging
    for (size_t i = 0; i < ellipsoidPoints.size(); i += 6) {
        std::cout << "Ellipsoid " << i / 6 << ": Position(" << ellipsoidPoints[i] << ", " << ellipsoidPoints[i + 1] << ", " << ellipsoidPoints[i + 2]
                  << "), Color(" << ellipsoidPoints[i + 3] << ", " << ellipsoidPoints[i + 4] << ", " << ellipsoidPoints[i + 5] << ")" << std::endl;
    }

    // // Create ellipsoid data (replace this with your ellipsoid data)
    // Eigen::MatrixXf ellipsoidMatrix = Eigen::MatrixXf::Random(10, 3);  // Adjust the number of ellipsoids as needed
    // ellipsoidPoints.resize(ellipsoidMatrix.size());
    // std::memcpy(ellipsoidPoints.data(), ellipsoidMatrix.data(), ellipsoidMatrix.size() * sizeof(float));

    // // Create ellipsoid data (replace this with your ellipsoid data)
    // Eigen::MatrixXf ellipsoidMatrix = Eigen::MatrixXf::Random(100, 3);  // Adjust the number of ellipsoids as needed
    // ellipsoidPoints.resize(ellipsoidMatrix.size());
    // std::memcpy(ellipsoidPoints.data(), ellipsoidMatrix.data(), ellipsoidMatrix.size() * sizeof(float));

    // Generate Vertex Buffer Object (VBO) for ellipsoids
    glGenBuffers(1, &vboEllipsoid);
    glBindBuffer(GL_ARRAY_BUFFER, vboEllipsoid);
    glBufferData(GL_ARRAY_BUFFER, ellipsoidPoints.size() * sizeof(GLfloat), ellipsoidPoints.data(), GL_STATIC_DRAW);

    // Generate Vertex Array Object (VAO) for ellipsoids
    glGenVertexArrays(1, &vaoEllipsoid);
    glBindVertexArray(vaoEllipsoid);

    // Specify the layout of the ellipsoid vertex data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Specify the layout of the color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    // Unbind VAO and VBO for ellipsoidsVAO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use the shader program
    glUseProgram(shaderProgram);

    // Get the uniform locations
    GLint projectionMatrixLocation = glGetUniformLocation(shaderProgram, "projectionMatrix");
    GLint viewMatrixLocation = glGetUniformLocation(shaderProgram, "viewMatrix");
    GLint modelMatrixLocation = glGetUniformLocation(shaderProgram, "modelMatrix");  // Declare modelMatrixLocation here

    // Pass matrices to shader if the uniform locations are valid
    if (projectionMatrixLocation != -1) {
        glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    } else {
        std::cerr << "Error: Projection matrix uniform not found in the shader." << std::endl;
    }

    if (viewMatrixLocation != -1) {
        glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    } else {
        std::cerr << "Error: View matrix uniform not found in the shader." << std::endl;
    }

    if (modelMatrixLocation != -1) {
        glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    } else {
        std::cerr << "Error: Model matrix uniform not found in the shader." << std::endl;
    }

    // Render ellipsoids
    glBindVertexArray(vaoEllipsoid);
    glDrawArrays(GL_POINTS, 0, ellipsoidPoints.size() / 6);
    glBindVertexArray(0);

    // Render points
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, 77777);
    glBindVertexArray(0);

    
    // Unbind the shader program
    glUseProgram(0);

    // Check for OpenGL errors after rendering
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error after rendering: " << error << std::endl;
    }
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
    glfwSetKeyCallback(window, keyCallback);
    // Set mouse callback
    glfwSetCursorPosCallback(window, mouse_callback);

    glewExperimental = GL_TRUE; // Enable experimental features for GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW initialization failed" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // Vertex shader source code
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;  // Color attribute

        uniform mat4 projectionMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 modelMatrix;  // Transformation matrix for ellipsoids

        out vec3 Color;  // Output color to fragment shader

        void main() {
            gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(aPos, 1.0);
            Color = aColor;  // Pass color to fragment shader
        }   
    )";


    // Fragment shader source code
    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 Color;  // Input color from vertex shader
        out vec4 FragColor;

        void main() {
            FragColor = vec4(Color, 1.0);
        }
    )";

    // Compile shaders
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
        GLint logLength;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            GLchar* log = new GLchar[logLength];
            glGetProgramInfoLog(shaderProgram, logLength, NULL, log);
            std::cerr << "Shader program linking failed: " << log << std::endl;
            delete[] log;
        }
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }


    // Matrix setup: Define a basic perspective projection matrix
    float fov = glm::radians(45.0f);  // Field of view in degrees
    float aspectRatio = 800.0f / 600.0f;  // Width divided by height
    float nearClip = 0.1f;
    float farClip = 100.0f;

    projectionMatrix = glm::perspective(fov, aspectRatio, nearClip, farClip);
    viewMatrix = glm::lookAt(glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // Setup VBO and VAO for points
    setupBuffers();

    // Setup VBO and VAO for ellipsoids
    setupEllipsoidBuffer();

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0f, 0.f, 0.0f, 1.0f); // Black background

        // Handle input
        processInput(window);

        // Update view matrix based on camera parameters
        viewMatrix = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);

        // Call the rendering function
        render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &vao);
    glDeleteVertexArrays(1, &vaoEllipsoid);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
