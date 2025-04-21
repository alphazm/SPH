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
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const int PARTICLE_COUNT = 600;
const float TIME_STEP = 0.0005f;
const float PARTICLE_RADIUS = 9.0f;
const float REST_DENSITY = 1000.0f;
const float GAS_CONSTANT = 5.0f;
const float VISCOSITY = 0.3f;
const float MASS = 0.05f;
const float H = 12.3f;
const float GRAVITY = -9.8f;

struct Vec2 {
    float x, y;
    Vec2(float _x = 0, float _y = 0) : x(_x), y(_y) {}

    Vec2 operator+(const Vec2& rhs) const { return Vec2(x + rhs.x, y + rhs.y); }
    Vec2 operator-(const Vec2& rhs) const { return Vec2(x - rhs.x, y - rhs.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    Vec2 operator/(float scalar) const { return Vec2(x / scalar, y / scalar); } // ✅ 修复的部分
    float length() const { return std::sqrt(x * x + y * y); }
    Vec2 normalize() const { float len = length(); return len > 0 ? Vec2(x / len, y / len) : Vec2(); }
};

struct Particle {
    Vec2 position, velocity, force;
    float density, pressure;
};

vector<Particle>MPI_particles;
Vec2 mousePos;
bool attract = false, repel = false;

void initParticles() {
    int side = std::sqrt(PARTICLE_COUNT); 
    float spacing = 12.0f;
    float gridWidth = (side - 1) * spacing; 
    float gridHeight = ((PARTICLE_COUNT + side - 1) / side - 1) * spacing; 
    float startX = (WINDOW_WIDTH - gridWidth) / 2.0f; 
    float startY = (WINDOW_HEIGHT - WINDOW_HEIGHT /2) ; 

    MPI_particles.resize(PARTICLE_COUNT);
    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        int x = i % side;
        int y = i / side;
        MPI_particles[i].position = Vec2(startX + x * spacing, startY - y * spacing);
        MPI_particles[i].velocity = Vec2(0, 0);
        MPI_particles[i].density = REST_DENSITY;
        MPI_particles[i].pressure = 0.0f;
        MPI_particles[i].force = Vec2(0, 0);
    }
}

void computeDensityPressure() {
    for (auto& pi :MPI_particles) {
        pi.density = 0;
        for (auto& pj :MPI_particles) {
            Vec2 rij = pj.position - pi.position;
            float r2 = rij.length();
            if (r2 < H) {
                float term = (H * H - r2 * r2);
                pi.density += MASS * term * term * term;
            }
        }
        pi.density *= 315.0f / (64.0f * M_PI * std::pow(H, 9));
        pi.pressure = GAS_CONSTANT * (pi.density - REST_DENSITY);
    }
}

void computeForces() {
    for (auto& pi :MPI_particles) {
        Vec2 pressureForce, viscosityForce;
        for (auto& pj :MPI_particles) {
            if (&pi == &pj) continue;
            Vec2 rij = pj.position - pi.position;
            float r = rij.length();
            if (r < H) {
                float wSpiky = -45.0f / (M_PI * std::pow(H, 6)) * std::pow(H - r, 2);
                pressureForce = pressureForce - rij.normalize() * MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * wSpiky;

                Vec2 velDiff = pj.velocity - pi.velocity;
                float wVisc = 45.0f / (M_PI * std::pow(H, 6)) * (H - r);
                viscosityForce = viscosityForce + velDiff * MASS * VISCOSITY / pj.density * wVisc;
            }
        }
        Vec2 gravityForce(0, MASS * GRAVITY);
        pi.force = pressureForce + viscosityForce + (gravityForce * 5.0f);
    }
}

void integrate(float dt) {
    for (auto& p :MPI_particles) {
        p.velocity = p.velocity + p.force * (dt / p.density);
        p.position = p.position + p.velocity * dt;

        // Boundary collisions
        if (p.position.x < PARTICLE_RADIUS) {
            p.position.x = PARTICLE_RADIUS;
            p.velocity.x *= -0.5f;
        }
        else if (p.position.x > WINDOW_WIDTH - PARTICLE_RADIUS) {
            p.position.x = WINDOW_WIDTH - PARTICLE_RADIUS;
            p.velocity.x *= -0.5f;
        }
        if (p.position.y < PARTICLE_RADIUS) {
            p.position.y = PARTICLE_RADIUS;
            p.velocity.y *= -0.5f;
        }
        else if (p.position.y > WINDOW_HEIGHT - PARTICLE_RADIUS) {
            p.position.y = WINDOW_HEIGHT - PARTICLE_RADIUS;
            p.velocity.y *= -0.5f;
        }

        if (attract || repel) {
            Vec2 dir = mousePos - p.position;
            float dist = dir.length();
            if (dist < 100.0f) {
                Vec2 forceDir = dir.normalize();
                float strength = (100.0f - dist) * (repel ? -100 : 100);
                p.velocity = p.velocity + forceDir * (strength * dt);
            }
        }
    }
}

void renderParticles() {
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(PARTICLE_RADIUS);
    glBegin(GL_POINTS);
    glColor3f(0.3f, 0.7f, 1.0f);
    for (auto& p :MPI_particles) {
        glVertex2f(p.position.x / (WINDOW_WIDTH / 2.0f) - 1.0, p.position.y / (WINDOW_HEIGHT / 2.0f) - 1.0);
    }
    glEnd();
}

static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    mousePos = Vec2(xpos, WINDOW_HEIGHT - ypos);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        attract = (action == GLFW_PRESS);
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        repel = (action == GLFW_PRESS);
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

        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);

        glClearColor(0, 0, 0, 1);

        initParticles();
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frames = 0;

        while (!glfwWindowShouldClose(window)) {
            computeDensityPressure();
            computeForces();
            integrate(TIME_STEP);

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
