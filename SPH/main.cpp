#include "main.h"
#include "openMP.h"
#include "MPI.h"
#include <mpi.h>
#include <iostream>
using namespace std;
int method = 0;;

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

//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);  // Initialize MPI once
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    int method = 0;
//    if (rank == 0) {
//        std::cout << "Select method:\n"
//            << "1. Serial\n"
//            << "2. OpenMP\n"
//            << "3. CUDA\n"
//            << "4. MPI\n"
//            << ": ";
//        std::cin >> method;
//    }
//    // Broadcast the choice to all ranks
//    MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//    switch (method) {
//    case 1:
//        if (rank == 0) std::cout << "Running Serial on rank 0 only.\n";
//        if (rank == 0) serial_main();
//        break;
//    case 2:
//        if (rank == 0) std::cout << "Running OpenMP.\n";
//        openMP_main();
//        break;
//    case 3:
//        if (rank == 0) std::cout << "Running CUDA.\n";
//        CUDA_main();
//        break;
//    case 4:
//        if (rank == 0) std::cout << "Running MPI version on all ranks.\n";
//        MPI_main();  // Call MPI_main without re-initializing MPI
//        break;
//    default:
//        if (rank == 0) std::cerr << "Invalid method.\n";
//    }
//
//    MPI_Finalize();  // Finalize MPI once
//    return 0;
//}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);  // Initialize MPI once
	int rank;
    float avg_ups[5] = {};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        cout << "serial" << endl;
        avg_ups[0]= serial_performance_test();
        cout << "openMP" << endl;
        avg_ups[1] = openMP_performance_test();
        cout << "CUDA" << endl;
        avg_ups[2] = CUDA_performance_test();

        cout << "MPI" << endl;
    }
	avg_ups[3] = MPI_performance_test();
        
    if (rank == 0) {
        cout << "Average Updates Per Second:" << endl;
        cout << "Serial: " << avg_ups[0] << endl;
        cout << "OpenMP: " << avg_ups[1] << endl;
        cout << "CUDA: " << avg_ups[2] << endl;
        cout << "MPI: " << avg_ups[3] << endl;
    }

	MPI_Finalize();  // Finalize MPI once
	return 0;
}