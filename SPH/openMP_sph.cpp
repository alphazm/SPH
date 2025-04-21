#include <windows.h>
#include <gl/GL.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <string>
#include <sstream> // Include for std::wstringstream
#include <cstdlib>

#pragma comment (lib, "OpenGL32.lib")

using namespace std;

// Constants
const int WIDTH = 500;
const int HEIGHT = 600;
const int NUM_PARTICLES = 500;
const float TIME_STEP = 0.003f;
const float PARTICLE_RADIUS = 10.0f;
const float REST_DENSITY = 1000.0f;
const float GAS_CONSTANT = 2000.0f;
const float VISCOSITY = 250.0f;
const float MASS = 65.0f;
const float H = 16.0f;
const float EPSILON = 1.0f;
const float GRAVITY_X = 0.0f;
const float GRAVITY_Y = -9.8f * 100.0f;
const float RESTITUTION = -0.8f; // Bounce factor

// [MOUSE]
const float MOUSE_FORCE = 500000.0f;
const float MOUSE_RADIUS = 500.0f;

float mouseX = 0.0f, mouseY = 0.0f;
bool mouseDown = false;

// Particle structure
struct Particle {
    float x, y;
    float vx, vy;
    float ax, ay;
    float density, pressure;
};

vector<Particle> particles(NUM_PARTICLES);

// SPH Kernel functions
float poly6(float r2) {
    float h2 = H * H;
    return (r2 >= 0 && r2 <= h2) ? (315.0f / (64.0f * 3.141592f * powf(H, 9))) * powf(h2 - r2, 3) : 0.0f;
}

float spiky(float r) {
    return (r > 0 && r <= H) ? (-45.0f / (3.141592f * pow(H, 6))) * pow(H - r, 2) : 0.0f;
}

float visco_lap(float r) {
    return (r >= 0 && r <= H) ? (45.0f / (3.141592f * pow(H, 6))) * (H - r) : 0.0f;
}

// Simulation update
void update_simulation() {
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle& pi = particles[i];
        pi.density = 0.0f;
        for (int j = 0; j < NUM_PARTICLES; ++j) {
            float dx = particles[j].x - pi.x;
            float dy = particles[j].y - pi.y;
            float r2 = dx * dx + dy * dy;
            pi.density += MASS * poly6(r2);
        }
        pi.pressure = GAS_CONSTANT * (pi.density - REST_DENSITY);
    }

#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle& pi = particles[i];
        pi.ax = 0.0f; pi.ay = 0.0f;
        for (int j = 0; j < NUM_PARTICLES; ++j) {
            if (i == j) continue;
            float dx = particles[j].x - pi.x;
            float dy = particles[j].y - pi.y;
            float r = sqrtf(dx * dx + dy * dy);
            if (r < EPSILON || r > H) continue;

            float pressure_term = -MASS * (pi.pressure + particles[j].pressure) / (2 * particles[j].density) * spiky(r);

            float force_x = pressure_term * dx / r;
            float force_y = pressure_term * dy / r;

            const float MAX_FORCE = 10000.0f;
            if (fabs(force_x) > MAX_FORCE) force_x = (force_x > 0) ? MAX_FORCE : -MAX_FORCE;
            if (fabs(force_y) > MAX_FORCE) force_y = (force_y > 0) ? MAX_FORCE : -MAX_FORCE;

            pi.ax += force_x;
            pi.ay += force_y;

            float visc = visco_lap(r);
            float visc_x = VISCOSITY * MASS * (particles[j].vx - pi.vx) / particles[j].density * visc;
            float visc_y = VISCOSITY * MASS * (particles[j].vy - pi.vy) / particles[j].density * visc;
            pi.ax += visc_x;
            pi.ay += visc_y;
        }

        pi.ax += GRAVITY_X;
        pi.ay += GRAVITY_Y;
    }

#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle& p = particles[i];

        // [MOUSE] Push effect
        if (mouseDown) {
            float mx = mouseX;
            float my = HEIGHT - mouseY;  // Flip Y to match OpenGL coords
            float dx = p.x - mx;
            float dy = p.y - my;
            float dist2 = dx * dx + dy * dy;

            if (dist2 < MOUSE_RADIUS * MOUSE_RADIUS && dist2 > 1.0f) {
                float dist = sqrtf(dist2);
                float force = MOUSE_FORCE / dist2;

                p.vx += (dx / dist) * force * TIME_STEP;
                p.vy += (dy / dist) * force * TIME_STEP;
            }
        }

        p.vx += TIME_STEP * p.ax;
        p.vy += TIME_STEP * p.ay;
        p.x += TIME_STEP * p.vx;
        p.y += TIME_STEP * p.vy;

        p.vx *= 0.995f;
        p.vy *= 0.995f;

        if (p.x < 0) {
            p.vx *= RESTITUTION;
            p.x = 0;
        }
        if (p.x > WIDTH) {
            p.vx *= RESTITUTION;
            p.x = WIDTH;
        }
        if (p.y < 0) {
            p.vy *= RESTITUTION;
            p.y = 0;
        }
        if (p.y > HEIGHT) {
            p.vy *= RESTITUTION;
            p.y = HEIGHT;
        }
    }
}

void draw_particles() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    for (const Particle& p : particles) {
        glVertex2f(p.x / (WIDTH / 2.0f) - 1.0f, p.y / (HEIGHT / 2.0f) - 1.0f);
    }
    glEnd();
}

class FPSCounter {
public:
    void startFrame() { start = chrono::high_resolution_clock::now(); }
    void endFrame(HWND hwnd) {
        auto end = chrono::high_resolution_clock::now();
        float fps = 1.0f / chrono::duration<float>(end - start).count();

        // Use std::wstringstream to construct the string
        std::wstringstream wss;
        wss << L"SPH Fluid Simulation - FPS: " << static_cast<int>(fps);
        SetWindowTextW(hwnd, wss.str().c_str());
    }
private:
    chrono::high_resolution_clock::time_point start;
};


LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_LBUTTONDOWN:
        mouseDown = true;
        break;
    case WM_LBUTTONUP:
        mouseDown = false;
        break;
    case WM_MOUSEMOVE:
        mouseX = LOWORD(lParam);
        mouseY = HIWORD(lParam);
        break;
    case WM_CLOSE:
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

int openMP_main() {
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    int nCmdShow = SW_SHOW;

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "SPHSim";
    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, wc.lpszClassName, "SPH Fluid Simulation", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WIDTH, HEIGHT, nullptr, nullptr, hInstance, nullptr);

    HDC hdc = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA, 32, 0,0,0,0,0,0,0,0,0,0,0,0, 24, 8, 0, PFD_MAIN_PLANE, 0, 0, 0, 0 };
    SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);

    HGLRC hglrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc, hglrc);

    ShowWindow(hwnd, nCmdShow);

    // Initialize particles in a centered grid
    int gridCols = (int)sqrt(NUM_PARTICLES);
    int gridRows = NUM_PARTICLES / gridCols;
    float spacing = PARTICLE_RADIUS * 1.5f;
    float offsetX = (WIDTH - gridCols * spacing) / 2.0f;
    float offsetY = (HEIGHT - gridRows * spacing) / 2.0f;
    int idx = 0;

    for (int i = 0; i < gridRows; ++i) {
        for (int j = 0; j < gridCols && idx < NUM_PARTICLES; ++j) {
            particles[idx].x = offsetX + j * spacing;
            particles[idx].y = offsetY + i * spacing;
            particles[idx].vx = particles[idx].vy = particles[idx].ax = particles[idx].ay = 0.0f;
            idx++;
        }
    }

    glPointSize(PARTICLE_RADIUS);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    MSG msg;
    FPSCounter fps;
    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        fps.startFrame();
        update_simulation();
        draw_particles();
        SwapBuffers(hdc);
        fps.endFrame(hwnd);
    }

	return 0;
}
