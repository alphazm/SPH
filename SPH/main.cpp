#include "main.h"
#include "MPI.h"
#include <iostream>
using namespace std;
int method = 0;

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1024; // Default value
    

    while (true) {
        if (rank == 0) {
            cout << "Enter number of particles (must be divisible by number of MPI processes for MPI method): ";
            cin >> N;
            if (N <= 0) {
                cerr << "Invalid number of particles. Using default N=8192.\n";
                N = 1024;
            }
        }
        // Broadcast N to all ranks
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int method = 0;
        if (rank == 0) {
            cout << "Select method:\n"
                << "1. Serial\n"
                << "2. OpenMP\n"
                << "3. CUDA\n"
                << "4. MPI\n"
                << ": ";
            cin >> method;
        }
        MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int mode = 0;
        if (rank == 0) {
            cout << "Select mode:\n"
                << "1. Visual output\n"
                << "2. Performance test\n"
                << ": ";
            cin >> mode;
            if (mode != 1 && mode != 2) {
                cerr << "Invalid mode. Defaulting to Visual output.\n";
                mode = 1;
            }
        }
        MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        switch (method) {
        case 1: // Serial
            if (rank == 0) {
                cout << "Running Serial with N=" << N << " on rank 0.\n";
                if (mode == 1) {
                    serial_main(N);
                }
                else {
                    float ups = serial_performance_test(N);
                    cout << "Serial Average UPS: " << ups << endl;
                }
            }
            break;
        case 2: // OpenMP
            if (rank == 0) {
                cout << "Running OpenMP with N=" << N << ".\n";
                if (mode == 1) {
                    openMP_main(N);
                }
                else {
                    float ups = openMP_performance_test(N);
                    cout << "OpenMP Average UPS: " << ups << endl;
                }
            }
            break;
        case 3: // CUDA
            if (rank == 0) {
                cout << "Running CUDA with N=" << N << ".\n";
                if (mode == 1) {
                    CUDA_main(N);
                }
                else {
                    float ups = CUDA_performance_test(N);
                    cout << "CUDA Average UPS: " << ups << endl;
                }
            }
            break;
        case 4: // MPI
            cout << "Running MPI with N=" << N << " on all ranks.\n";
            if (mode == 1) {
                MPI_main(N);
            }
            else {
                float ups = MPI_performance_test(N);
                if (rank == 0) {
                    cout << "MPI Average UPS: " << ups << endl;
                }
            }
            break;
        default:
            if (rank == 0) cerr << "Invalid method.\n";
        }
    }

    MPI_Finalize();
    return 0;
}

//int main(int argc, char** argv) {
//	MPI_Init(&argc, &argv);  // Initialize MPI once
//	int rank;
//    float avg_ups[5] = {};
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    
//    if (rank == 0) {
//        cout << "serial" << endl;
//        //avg_ups[0]= serial_performance_test();
//        cout << "openMP" << endl;
//        //avg_ups[1] = openMP_performance_test();
//        cout << "CUDA" << endl;
//        //avg_ups[2] = CUDA_performance_test();
//
//        cout << "MPI" << endl;
//    }
//	avg_ups[3] = MPI_performance_test();
//        
//    if (rank == 0) {
//        cout << "Average Updates Per Second:" << endl;
//        cout << "Serial: " << avg_ups[0] << endl;
//        cout << "OpenMP: " << avg_ups[1] << endl;
//        cout << "CUDA: " << avg_ups[2] << endl;
//        cout << "MPI: " << avg_ups[3] << endl;
//    }
//
//	MPI_Finalize();  // Finalize MPI once
//	return 0;
//}