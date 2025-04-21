#include "main.h"
#include "openMP.h"
#include "MPI.h"
#include <iostream>

void errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;
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

double mouseX = 0.0, mouseY = 0.0;
int mouseButtonState = 0;

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    mouseX = xpos;
    mouseY = ypos;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        mouseButtonState = 1;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        mouseButtonState = 2;
    else
        mouseButtonState = 0;
}

int main(int argc, char** argv){
    //serial_main();
	//CUDA_main();
	//openMP_main();
	MPI_main(argc, argv);
	return 0;
}