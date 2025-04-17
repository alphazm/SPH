#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "sph.cuh"

using namespace std;

void errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;
}

constexpr float PARTICLE_SIZE = 8.0f;

// 1 = pixel on, 0 = pixel off
const unsigned char font5x7[15][7] = {
    // 0
    {0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F},
    // 1
    {0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x1F},
    // 2
    {0x1F, 0x10, 0x10, 0x1F, 0x01, 0x01, 0x1F},
    // 3
    {0x1F, 0x10, 0x10, 0x1F, 0x10, 0x10, 0x1F},
    // 4
    {0x11, 0x11, 0x11, 0x1F, 0x10, 0x10, 0x10},
    // 5
    {0x1F, 0x01, 0x01, 0x1F, 0x10, 0x10, 0x1F},
    // 6
    {0x1F, 0x01, 0x01, 0x1F, 0x11, 0x11, 0x1F},
    // 7
    {0x1F, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10},
    // 8
    {0x1F, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x1F},
    // 9
    {0x1F, 0x11, 0x11, 0x1F, 0x10, 0x10, 0x1F},
    // .
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06},
    // F
    {0x1F, 0x01, 0x01, 0x1F, 0x01, 0x01, 0x01},
    // P
    {0x1F, 0x11, 0x11, 0x1F, 0x01, 0x01, 0x01},
    // S
    {0x1F, 0x01, 0x01, 0x1F, 0x10, 0x10, 0x1F},
    // :
    {0x00, 0x06, 0x06, 0x00, 0x06, 0x06, 0x00}
};

double mouseX, mouseY; int mouseButtonState = 0;

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    mouseX = xpos; mouseY = ypos; 
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) { 
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) 
        mouseButtonState = 1; 
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) 
        mouseButtonState = 2; 
    else mouseButtonState = 0; 
	
}

// Render a character at (x, y) with scale
void renderChar(float x, float y, char c, float scale) {
    int charIndex;
    if (c >= '0' && c <= '9') charIndex = c - '0';
    else if (c == '.') charIndex = 10;
    else if (c == 'F') charIndex = 11;
    else if (c == 'P') charIndex = 12;
    else if (c == 'S') charIndex = 13;
    else if (c == ':') charIndex = 14;
    else return; // Unsupported character

    glPointSize(2.0f); // Small points for pixels
    glBegin(GL_POINTS);
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (font5x7[charIndex][6 - row] & (1 << col)) { // Flip row for OpenGL
                float px = x + col * scale;
                float py = y + row * scale;
                glVertex2f(px, py);
            }
        }
    }
    glEnd();
}

// Render string at (x, y)
void renderString(float x, float y, const char* str, float scale) {
    float cx = x;
    while (*str) {
        renderChar(cx, y, *str, scale);
        cx += 6 * scale; // Character width + spacing
        str++;
    }
}

int main() {
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 800, "SPH Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window" << std::endl;
        return -1;
    }
    glfwMakeContextCurrent(window);
    glViewport(0, 0, 800, 800);

    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
  

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Initialize CUDA-OpenGL interop
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    cudaSetDevice(0);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float2), NULL, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cudaVBO;
    cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsWriteDiscard);

    Particle* particles = new Particle[N];
    initSimulation(particles, cudaVBO);

    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    //glPointSize(PARTICLE_SIZE); // Use defined particle size
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 2, 0, 2, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    // FPS counter variables
    double lastTime = glfwGetTime();
    int frameCount = 0;
    double fps = 0.0;

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        float simX = (mouseX / 800.0) * 2.0;
        float simY = 2.0 - (mouseY / 800.0) * 2.0;
        float2 mousePos = make_float2(simX, simY);
        float interactionStrength = (mouseButtonState == 1) ? 50000.0f : (mouseButtonState == 2) ? -50000.0f : 0.0f;
    
		// Update simulation
        stepSimulation(particles, nullptr, nullptr, nullptr, cudaVBO, mousePos, interactionStrength);

        // Update FPS
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            fps = frameCount / (currentTime - lastTime);
            frameCount = 0;
            lastTime = currentTime;
        }
        glPointSize(PARTICLE_SIZE);
        // Render particles
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(2, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glColor3f(0.0f, 0.5f, 1.0f);
        glDrawArrays(GL_POINTS, 0, N);
        glDisableClientState(GL_VERTEX_ARRAY);

        // Render FPS text
        glColor3f(1.0f, 1.0f, 1.0f);
        char fpsText[16];
        snprintf(fpsText, sizeof(fpsText), "FPS: %.0f", fps);
        renderString(0.05f, 1.9f, fpsText, 0.01f);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
    delete[] particles;
	cleanupSimulation();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}