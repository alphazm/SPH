// mpi_sph_liquid_sim.cpp

#include <mpi.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include "MPI.h"
#include "main.h"

using namespace std;

#define M_PI 3.141596
// Simulation parameters
#define N 500
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
#define PARTICLE_RADIUS 8.0f
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

struct Vec2 {
    float x, y;
    Vec2(float _x = 0, float _y = 0) : x(_x), y(_y) {}
    Vec2 operator+(const Vec2& rhs) const { return Vec2(x + rhs.x, y + rhs.y); }
    Vec2 operator-(const Vec2& rhs) const { return Vec2(x - rhs.x, y - rhs.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    Vec2 operator/(float scalar) const { return Vec2(x / scalar, y / scalar); }
    float length() const { return std::sqrt(x * x + y * y); }
    Vec2 normalize() const { float len = length(); return len > 0 ? Vec2(x / len, y / len) : Vec2(); }
};

struct Particle {
    Vec2 position, velocity;
    float density, pressure;
    bool valid;
};

vector<Particle> MPI_particles;
Vec2 mousePos;
float mpi_interactionStrength = 0.0f;

// Serial-aligned kernel functions
float mpi_poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 > h2) return 0.0f;
    float term = h2 - r2;
    return (315.0f / (64.0f * M_PI * powf(h, 9))) * term * term * term;
}

Vec2 spikyGrad(Vec2 r, float r_len, float h) {
    if (r_len > h || r_len < 1e-6) return Vec2(0, 0);
    float term = h - r_len;
    float coeff = -45.0f / (M_PI * powf(h, 6)) * term * term / r_len;
    return Vec2(coeff * r.x, coeff * r.y);
}

float mpi_viscosityLaplacian(float r, float h) {
    if (r > h) return 0.0f;
    return 45.0f / (M_PI * powf(h, 6)) * (h - r);
}

Vec2 ljForce(Vec2 r, float r_len, float sigma, float epsilon) {
    if (r_len >= 2.5f * sigma || r_len < 1e-6f) return Vec2(0, 0);
    float inv_r = 1.0f / r_len;
    float sigma_over_r = sigma * inv_r;
    float sigma_over_r2 = sigma_over_r * sigma_over_r;
    float sigma_over_r6 = sigma_over_r2 * sigma_over_r2 * sigma_over_r2;
    float sigma_over_r12 = sigma_over_r6 * sigma_over_r6;
    float inv_r2 = inv_r * inv_r;
    float coeff = 24.0f * epsilon * (2.0f * sigma_over_r12 - sigma_over_r6) * inv_r2;
    return Vec2(coeff * r.x, coeff * r.y);
}

void mpi_initParticles() {
    MPI_particles.resize(N);
    for (int i = 0; i < N; i++) {
        MPI_particles[i].position = Vec2(0.7f + 0.05f * (i % 20), 1.5f - 0.05f * (i / 20));
        MPI_particles[i].velocity = Vec2(0.0f, 0.0f);
        MPI_particles[i].density = REST_DENSITY;
        MPI_particles[i].pressure = 0.0f;
        MPI_particles[i].valid = true;
    }
}

void computeDensityPressure() {
    for (auto& pi : MPI_particles) {
        pi.density = 0.0f;
        for (auto& pj : MPI_particles) {
            Vec2 r = pj.position - pi.position;
            float r2 = r.length() * r.length();
            if (r2 < H * H) {
                pi.density += MASS * mpi_poly6(r2, H);
            }
        }
        pi.pressure = STIFFNESS * (pi.density - REST_DENSITY);
        if (pi.pressure < 0.0f) pi.pressure = 0.0f;
    }
}

void computeForces() {
    for (auto& pi : MPI_particles) {
        if (!pi.valid) continue;
        Vec2 force(0.0f, 0.0f);
        for (auto& pj : MPI_particles) {
            if (&pi == &pj || !pj.valid) continue;
            Vec2 r = pi.position - pj.position;
            float r2 = r.length() * r.length();
            if (r2 >= H * H) continue;
            float r_len = sqrtf(r2);
            Vec2 grad = spikyGrad(r, r_len, H);
            float pressureTerm = (pi.pressure + pj.pressure) / (2.0f * pj.density);
            Vec2 pressureForce = Vec2(-MASS * pressureTerm * grad.x, -MASS * pressureTerm * grad.y);
            Vec2 relVel = pj.velocity - pi.velocity;
            float viscForce = VISCOSITY * MASS * mpi_viscosityLaplacian(r_len, H) / pj.density;
            Vec2 ljF = ljForce(r, r_len, SIGMA, EPSILON);
            force.x += pressureForce.x + viscForce * relVel.x + ljF.x;
            force.y += pressureForce.y + viscForce * relVel.y + ljF.y;
        }
        force.y -= 10.0f * pi.density;
        if (mpi_interactionStrength != 0.0f) {
            Vec2 r = pi.position - mousePos;
            float r2 = r.length() * r.length();
            if (r2 < 0.25f * 0.25f) {
                float r_len = sqrtf(r2);
                if (r_len > 1e-6f) {
                    float forceMag = mpi_interactionStrength * (0.25f - r_len) / 0.25f;
                    force.x += forceMag * r.x / r_len;
                    force.y += forceMag * r.y / r_len;
                }
            }
        }
        float forceMag = sqrtf(force.x * force.x + force.y * force.y);
        if (forceMag > 1000.0f * pi.density) {
            float scale = 1000.0f * pi.density / forceMag;
            force.x *= scale;
            force.y *= scale;
        }
        pi.velocity.x += DT * force.x / pi.density;
        pi.velocity.y += DT * force.y / pi.density;
    }
}

void integrate() {
    for (auto& p : MPI_particles) {
        if (!p.valid) continue;
        p.position.x += DT * p.velocity.x;
        p.position.y += DT * p.velocity.y;
        float velMag = sqrtf(p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y);
        if (velMag > MAX_VEL) {
            float scale = MAX_VEL / velMag;
            p.velocity.x *= scale;
            p.velocity.y *= scale;
        }
        if (p.position.x < 0.05f) { p.position.x = 0.05f; p.velocity.x = fabsf(p.velocity.x) * DAMPING; }
        if (p.position.x > 1.95f) { p.position.x = 1.95f; p.velocity.x = -fabsf(p.velocity.x) * DAMPING; }
        if (p.position.y < 0.05f) { p.position.y = 0.05f; p.velocity.y = fabsf(p.velocity.y) * DAMPING; }
        if (p.position.y > 1.95f) { p.position.y = 1.95f; p.velocity.y = -fabsf(p.velocity.y) * DAMPING; }
    }
}

void renderParticles() {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(PARTICLE_RADIUS);
    glBegin(GL_POINTS);
    glColor3f(0.3f, 0.7f, 1.0f);
    for (auto& p : MPI_particles) {
        glVertex2f(p.position.x, p.position.y); 
    }
    glEnd();
}

static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    mousePos = Vec2(xpos / WINDOW_WIDTH, 1.0f - ypos / WINDOW_HEIGHT);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) mpi_interactionStrength = 50000.0f;
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) mpi_interactionStrength = 0.0f;
}

int MPI_main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (!glfwInit()) return -1;
        GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "MPI SPH Fluid", NULL, NULL);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);
        glewInit();

        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 2, 0, 2, -1, 1); // Match serial version
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);

        glClearColor(0, 0, 0, 1);

        mpi_initParticles();
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frames = 0;

        while (!glfwWindowShouldClose(window)) {
            computeDensityPressure();
            computeForces();
            integrate();

            renderParticles();
            glfwSwapBuffers(window);
            glfwPollEvents();

            frames++;
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - lastTime).count();
            if (elapsed >= 1.0f) {
                std::cout << "FPS: " << frames << std::endl;
                frames = 0;
                lastTime = now;
            }
        }
        glfwTerminate();
    }
    MPI_Finalize();
    return 0;
}
