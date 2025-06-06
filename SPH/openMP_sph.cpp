#include <windows.h>
#include <gl/GL.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <string>
#include <cstdlib>
#include <DirectXMath.h>
#include <iostream>

#pragma comment (lib, "OpenGL32.lib")

using namespace std;
using namespace DirectX;


#define H 0.15f
#define DT 0.0005f
#define MASS 0.05f
#define VISCOSITY 0.3f
#define STIFFNESS 1000.0f
#define REST_DENSITY 1000.0f
#define DAMPING 0.9f
#define MAX_VEL 5.0f
#define SIGMA 0.05f
#define EPSILON 1.0f
#define WIDTH 800
#define HEIGHT 800
#define PARTICLE_RADIUS 8.0f

float avgFPS = 0;
int totalFrame = 0;

struct Particle {
    XMFLOAT2 pos;
    XMFLOAT2 vel;
    float density;
    float pressure;
    bool valid;
};


XMFLOAT2 mousePos = { 0.0f, 0.0f };
float interactionStrength = 0.0f;
bool isPushing = true;

// Kernel functions
inline float openMP_poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 > h2) return 0.0f;
    float term = h2 - r2;
    return (315.0f / (64.0f * 3.14159f * powf(h, 9))) * term * term * term;
}

inline XMFLOAT2 spikyGrad(XMFLOAT2 r, float r_len, float h) {
    if (r_len > h || r_len < 1e-6f) return { 0.0f, 0.0f };
    float coeff = -45.0f / (3.14159f * powf(h, 6)) * powf(h - r_len, 2) / r_len;
    return { coeff * r.x, coeff * r.y };
}

inline float openMP_viscosityLaplacian(float r, float h) {
    if (r > h) return 0.0f;
    return 45.0f / (3.14159f * powf(h, 6)) * (h - r);
}

inline XMFLOAT2 ljForce(XMFLOAT2 r, float r_len, float sigma, float epsilon) {
    if (r_len >= 2.5f * sigma || r_len < 1e-6f) return { 0.0f, 0.0f };
    float inv_r = 1.0f / r_len;
    float sigma_over_r = sigma * inv_r;
    float sigma_over_r2 = sigma_over_r * sigma_over_r;
    float sigma_over_r6 = sigma_over_r2 * sigma_over_r2 * sigma_over_r2;
    float sigma_over_r12 = sigma_over_r6 * sigma_over_r6;
    float inv_r2 = inv_r * inv_r;
    float coeff = 24.0f * epsilon * (2.0f * sigma_over_r12 - sigma_over_r6) * inv_r2;
    return { coeff * r.x, coeff * r.y };
}

// Physics
void computeDensityPressureOMP(vector<Particle>& particles, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Particle& p = particles[i];
        p.density = 0.0f;
        for (int j = 0; j < N; ++j) {
            XMFLOAT2 r = { p.pos.x - particles[j].pos.x, p.pos.y - particles[j].pos.y };
            float r2 = r.x * r.x + r.y * r.y;
            if (r2 < H * H) {
                p.density += MASS * openMP_poly6(r2, H);
            }
        }
        p.pressure = STIFFNESS * (p.density - REST_DENSITY);
        if (p.pressure < 0.0f) p.pressure = 0.0f;
    }
}

void computeForcesOMP(vector<Particle>& particles, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Particle& p = particles[i];
        if (!p.valid) continue;

        XMFLOAT2 force = { 0.0f, 0.0f };

        for (int j = 0; j < N; ++j) {
            if (i == j || !particles[j].valid) continue;
            XMFLOAT2 r = { p.pos.x - particles[j].pos.x, p.pos.y - particles[j].pos.y };
            float r2 = r.x * r.x + r.y * r.y;
            if (r2 >= H * H) continue;
            float r_len = sqrtf(r2);
            XMFLOAT2 grad = spikyGrad(r, r_len, H);
            float pressureTerm = (p.pressure + particles[j].pressure) / (2.0f * particles[j].density);
            XMFLOAT2 pressureForce = { -MASS * pressureTerm * grad.x, -MASS * pressureTerm * grad.y };
            XMFLOAT2 relVel = { particles[j].vel.x - p.vel.x, particles[j].vel.y - p.vel.y };
            float viscForce = VISCOSITY * MASS * openMP_viscosityLaplacian(r_len, H) / particles[j].density;
            XMFLOAT2 ljF = ljForce(r, r_len, SIGMA, EPSILON);
            force.x += pressureForce.x + viscForce * relVel.x + ljF.x;
            force.y += pressureForce.y + viscForce * relVel.y + ljF.y;
        }

        force.y -= 10.0f * p.density;

        // Mouse interaction
        if (interactionStrength != 0.0f) {
            XMFLOAT2 r = { p.pos.x - mousePos.x, p.pos.y - mousePos.y };
            float r2 = r.x * r.x + r.y * r.y;
            if (r2 < 0.25f * 0.25f) {
                float r_len = sqrtf(r2);
                if (r_len > 1e-6f) {
                    float forceMag = interactionStrength * (0.25f - r_len) / 0.25f;
                    force.x += forceMag * r.x / r_len;
                    force.y += forceMag * r.y / r_len;
                }
            }
        }

        float forceMag = sqrtf(force.x * force.x + force.y * force.y);
        if (forceMag > 1000.0f * p.density) {
            float scale = 1000.0f * p.density / forceMag;
            force.x *= scale;
            force.y *= scale;
        }

        p.vel.x += DT * force.x / p.density;
        p.vel.y += DT * force.y / p.density;
    }
}

void initParticles(vector<Particle>& particles, int N) {
    particles.resize(N);

    // Calculate grid dimensions
    int cols = static_cast<int>(std::ceil(std::sqrt(N))); // Number of columns
    int rows = (N + cols - 1) / cols; // Number of rows, rounded up
    if (rows == 0) rows = 1; // Ensure at least one row for small N

    // Define safe simulation range [0.05, 1.95]
    const float minPos = 0.05f;
    const float maxPos = 1.95f;
    const float range = maxPos - minPos; // 1.9

    // Calculate spacing to fit particles within [0.05, 1.95]
    float spacingX = range / (cols > 1 ? cols - 1 : 1); // Avoid division by zero
    float spacingY = range / (rows > 1 ? rows - 1 : 1);
    float spacing = min(spacingX, spacingY); // Use smaller spacing to avoid overlap

    // Center the grid
    float startX = minPos + (range - (cols - 1) * spacing) / 2.0f;
    float startY = maxPos - (range - (rows - 1) * spacing) / 2.0f;

    // Initialize particles
    for (int i = 0; i < N; i++) {
        int x = i % cols; // Column index
        int y = i / cols; // Row index
        particles[i].pos = { startX + x * spacing, startY - y * spacing };
        particles[i].vel = { 0.0f, 0.0f };
        particles[i].density = REST_DENSITY;
        particles[i].pressure = 0.0f;
        particles[i].valid = true;
    }
}

void integrateOMP(vector<Particle>& particles, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Particle& p = particles[i];

        //p.vel.y -= 9.81f * DT;
        if (!p.valid) continue;
        p.pos.x += DT * p.vel.x;
        p.pos.y += DT * p.vel.y;

        float velMag = sqrtf(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        if (velMag > MAX_VEL) {
            float scale = MAX_VEL / velMag;
            p.vel.x *= scale;
            p.vel.y *= scale;
        }

        if (p.pos.x < 0.05f) { p.pos.x = 0.05f; p.vel.x = fabsf(p.vel.x) * DAMPING; }
        if (p.pos.x > 1.95f) { p.pos.x = 1.95f; p.vel.x = -fabsf(p.vel.x) * DAMPING; }
        if (p.pos.y < 0.05f) { p.pos.y = 0.05f; p.vel.y = fabsf(p.vel.y) * DAMPING; }
        if (p.pos.y > 1.95f) { p.pos.y = 1.95f; p.vel.y = -fabsf(p.vel.y) * DAMPING; }
    }
}

void display(const vector<Particle>& particles) {
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable point smoothing for nicer points
    glEnable(GL_POINT_SMOOTH);
    glPointSize(PARTICLE_RADIUS);

    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        if (p.valid) {
            // Map coordinates to screen space
            float pressure_normalized = min(1.0f, p.pressure / (STIFFNESS * REST_DENSITY * 0.5f));
            glColor3f(0.2f + 0.8f * pressure_normalized, 0.6f + 0.4f * pressure_normalized, 1.0f);
            glVertex2f(p.pos.x, p.pos.y);
        }
    }
    glEnd();

    glDisable(GL_POINT_SMOOTH);
}

void update(vector<Particle>& particles, int N) {
    computeDensityPressureOMP(particles, N);
    computeForcesOMP(particles, N);
    integrateOMP(particles, N);
}

// Windows + OpenGL Boilerplate
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CLOSE: PostQuitMessage(0); cout << "Average FPS: "<< avgFPS/totalFrame << endl; return 0;
    case WM_MOUSEMOVE: {
        int x = LOWORD(lParam);
        int y = HIWORD(lParam);
        mousePos = { x / (float)WIDTH, (1.0f - y / (float)HEIGHT) * 1.5f };
        return 0;
    }
    case WM_LBUTTONDOWN: interactionStrength = isPushing ? 50000.0f : -5000.0f; return 0;
    case WM_LBUTTONUP: interactionStrength = 0.0f; return 0;
    case WM_RBUTTONDOWN: isPushing = !isPushing; return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int openMP_main(int N) {
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    int nCmdShow = SW_SHOW;

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "SPHWindow";
    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, "SPHWindow", "SPH Fluid Simulation", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WIDTH, HEIGHT, nullptr, nullptr, hInstance, nullptr);
    ShowWindow(hwnd, nCmdShow);

    HDC hdc = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER, PFD_TYPE_RGBA, 32 };
    SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
    HGLRC hglrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc, hglrc);

    // Set up OpenGL projection and viewport
    glViewport(0, 0, WIDTH, HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 2, 0, 2, -1, 1); // Match serial version
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Removed glOrtho
    vector<Particle> particles;
    initParticles(particles, N);

    MSG msg;
    auto lastTime = chrono::high_resolution_clock::now();
    int frameCount = 0;
    float FPS = 0.0f;


    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        update(particles, N);
        display(particles);
        SwapBuffers(hdc);

        frameCount++;
        auto currentTime = chrono::high_resolution_clock::now();
        chrono::duration<float> elapsed = currentTime - lastTime;
        if (elapsed.count() >= 1.0f) {
            FPS = frameCount / elapsed.count();
			avgFPS += FPS;
			totalFrame++;
            frameCount = 0;
            lastTime = currentTime;

            char title[256];
            sprintf_s(title, "SPH Fluid Simulation - Avg FPS: %.2f | Mode: %s", FPS, isPushing ? "Push" : "Pull");
            SetWindowTextA(hwnd, title);
        }
    }
	

    return 0;
}

float openMP_performance_test(int N) {
    std::vector<Particle> particles(N);
    initParticles(particles, N); // Use existing initialization

    int num_steps = 100;
    int num_runs = 10;
    double total_ups = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        auto start_time = std::chrono::high_resolution_clock::now();
		cout << "Run: " << run + 1 << " / " << num_runs << endl;
        for (int step = 0; step < num_steps; ++step) {
            computeDensityPressureOMP(particles, N);
            computeForcesOMP(particles, N);
            integrateOMP(particles, N);
			cout << "\nStep: " << step + 1 << " / " << num_steps << endl;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double ups = num_steps / elapsed.count();
        total_ups += ups;
    }

    float avg_ups = total_ups / num_runs;
	return avg_ups;
}