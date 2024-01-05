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

GLuint vbo; // Vertex Buffer Object
GLuint vao; // Vertex Array Object

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


GLuint compileShader(GLenum shaderType, const char* shaderSource) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, nullptr);
    glCompileShader(shader);

    // Check shader compilation status
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
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

void setupBuffers() {
    // Define the points in model space
    // simple vector of points
    // std::vector<GLfloat> points = {
    //     0.0f, 0.0f, 0.0f,
    //     0.0f, 0.0f, 1.0f,
    //     0.0f, 0.0f, 2.0f
    // };
    // Create an n by 3 matrix of random floats
    Eigen::MatrixXf eigenMatrix = Eigen::MatrixXf::Random(77777, 3);
    // Print the Eigen matrix
    std::cout << "Eigen Matrix:\n" << eigenMatrix << std::endl;

    // new
    // Convert Eigen matrix to a 10 by 3 std::vector
    std::vector<GLfloat> points(eigenMatrix.data(), eigenMatrix.data() + eigenMatrix.size());
    // Print the converted std::vector
    std::cout << "Converted std::vector:\n";
    for (size_t i = 0; i < points.size(); ++i) {
        std::cout << points[i] << " ";
        if ((i + 1) % 3 == 0) {
            std::cout << std::endl;
        }
    }

    // // old
    // // Convert Eigen matrix to std::vector
    // std::vector<GLfloat> points(eigenMatrix.data(), eigenMatrix.data() + eigenMatrix.size());
    // // Print the converted std::vector
    // std::cout << "points:\n";
    // for (const auto& value : points) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;

    // Generate Vertex Buffer Object (VBO)
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), points.data(), GL_STATIC_DRAW);

    // Generate Vertex Array Object (VAO)
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Specify the layout of the vertex data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Unbind VAO and VBO
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

    // Bind VAO
    glBindVertexArray(vao);

    //test
    // Get the size of the allocated storage
    // GLint bufferSize;
    // glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufferSize);
    // std::cout << "Size of VBO: " << bufferSize << " bytes" << std::endl;

    // Render points
    glDrawArrays(GL_POINTS, 0, 77777); // TODO: need to adjust this to the length of buffer

    // std::cout << "Size of std::array: " << sizeof(vao) << " bytes" << std::endl;

    // Unbind VAO
    glBindVertexArray(0);

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
            FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    )";

    // Compile shaders
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

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
    float farClip = 100.0f;

    projectionMatrix = glm::perspective(fov, aspectRatio, nearClip, farClip);
    viewMatrix = glm::lookAt(glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // Setup VBO and VAO
    setupBuffers();

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background

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
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
