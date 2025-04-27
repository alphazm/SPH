// Parallel SPH simulation using MPI (MPI_Allgather), modified to follow serial SPH code

#include <mpi.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

#define M_PI 3.141596f
// Simulation parameters 

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
    Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
};

struct Particle {
    Vec2 position, velocity;
    float density, pressure;
    bool valid;
};

// Mouse input globals (used on rank 0)
static double mouseX = 0.0, mouseY = 0.0;
static int mouseButtonState = 0;

// Mouse callbacks 
static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    mouseX = xpos;
    mouseY = ypos;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        mouseButtonState = 1;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        mouseButtonState = 2;
    else
        mouseButtonState = 0;
}

// Kernel functions 
float mpi_poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 > h2) return 0.0f;
    float term = h2 - r2;
    return (315.0f / (64.0f * 3.14159f * powf(h, 9))) * term * term * term;
}

Vec2 spikyGrad(const Vec2& r, float r_len, float h) {
    if (r_len > h || r_len < 1e-6f) return Vec2(0, 0);
    float term = h - r_len;
    float coeff = -45.0f / (3.14159f * powf(h, 6)) * term * term / r_len;
    return Vec2(coeff * r.x, coeff * r.y);
}

float mpi_viscosityLaplacian(float r, float h) {
    if (r > h) return 0.0f;
    return 45.0f / (3.14159f * powf(h, 6)) * (h - r);
}

Vec2 ljForce(const Vec2& r, float r_len, float sigma, float epsilon) {
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

// Particle initialization 
void mpi_initParticles(vector<Particle>& P, int N) {
    if (P.size() != N) P.resize(N);
    int cols = static_cast<int>(ceil(sqrtf(static_cast<float>(N))));
    int rows = (N + cols - 1) / cols;
    if (rows == 0) rows = 1;
    const float minPos = 0.05f;
    const float maxPos = 1.95f;
    const float range = maxPos - minPos;
    float spacingX = range / (cols > 1 ? cols - 1 : 1);
    float spacingY = range / (rows > 1 ? rows - 1 : 1);
    float spacing = min(spacingX, spacingY);
    float startX = minPos + (range - (cols - 1) * spacing) / 2.0f;
    float startY = maxPos - (range - (rows - 1) * spacing) / 2.0f;

    for (int i = 0; i < N; i++) {
        int x = i % cols;
        int y = i / cols;
        P[i].position = Vec2(startX + x * spacing, startY - y * spacing);
        P[i].velocity = Vec2(0, 0);
        P[i].density = REST_DENSITY;
        P[i].pressure = 0;
        P[i].valid = true;
    }
}

// Rendering function
void renderParticles(const vector<Particle>& P) {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(PARTICLE_RADIUS);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.5f, 1.0f); // Match serial color
    for (const auto& p : P) {
        if (p.valid && !isnan(p.position.x) && !isnan(p.position.y)) {
            glVertex2f(p.position.x, p.position.y);
        }
    }
    glEnd();
}

void mpi_computeStep(vector<Particle>& particles,int N, int start, int end, int N_local,
    vector<Vec2>& localPos, vector<Vec2>& localVel, vector<Vec2>& allPos, vector<Vec2>& allVel,
    vector<float>& localDens, vector<float>& localPres, vector<float>& allDens, vector<float>& allPres,
    const Vec2& mousePos, float interactionStrength) {
    // 1) Gather positions & velocities
    for (int i = 0; i < N_local; ++i) {
        localPos[i] = particles[start + i].position;
        localVel[i] = particles[start + i].velocity;
    }
    MPI_Allgather(localPos.data(), N_local * sizeof(Vec2), MPI_BYTE,
        allPos.data(), N_local * sizeof(Vec2), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(localVel.data(), N_local * sizeof(Vec2), MPI_BYTE,
        allVel.data(), N_local * sizeof(Vec2), MPI_BYTE, MPI_COMM_WORLD);

    // 2) Compute density & pressure (local, matches serial_computeDensityPressure)
    for (int idx = start; idx < end; ++idx) {
        float dens = 0;
        for (int j = 0; j < N; ++j) {
            Vec2 r = particles[idx].position - allPos[j];
            float r2 = r.x * r.x + r.y * r.y;
            if (r2 < H * H) dens += MASS * mpi_poly6(r2, H);
        }
        localDens[idx - start] = dens;
        particles[idx].density = dens;
        float p = STIFFNESS * (dens - REST_DENSITY);
        localPres[idx - start] = (p > 0 ? p : 0);
        particles[idx].pressure = localPres[idx - start];
    }
    MPI_Allgather(localDens.data(), N_local, MPI_FLOAT,
        allDens.data(), N_local, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(localPres.data(), N_local, MPI_FLOAT,
        allPres.data(), N_local, MPI_FLOAT, MPI_COMM_WORLD);

    // 3) Compute forces & integrate (local, matches serial_computeForces and serial_integrate)
    for (int idx = start; idx < end; ++idx) {
        if (!particles[idx].valid) continue;
        Vec2 force(0, 0);
        for (int j = 0; j < N; ++j) {
            if (j == idx || !particles[j].valid) continue;
            Vec2 dr = particles[idx].position - allPos[j];
            float r2 = dr.x * dr.x + dr.y * dr.y;
            if (r2 >= H * H) continue;
            float r_len = sqrtf(r2);
            Vec2 grad = spikyGrad(dr, r_len, H);
            float pt = (particles[idx].pressure + allPres[j]) / (2 * allDens[j]);
            Vec2 pForce = Vec2(-MASS * pt * grad.x, -MASS * pt * grad.y);
            Vec2 rv = allVel[j] - particles[idx].velocity;
            float visc = VISCOSITY * MASS * mpi_viscosityLaplacian(r_len, H) / allDens[j];
            Vec2 vForce = Vec2(visc * rv.x, visc * rv.y);
            Vec2 lj = ljForce(dr, r_len, SIGMA, EPSILON);
            force.x += pForce.x + vForce.x + lj.x;
            force.y += pForce.y + vForce.y + lj.y;
        }
        force.y -= 10.0f * particles[idx].density; // Gravity

        // Add mouse interaction (matches serial_computeForces)
        if (interactionStrength != 0.0f) {
            Vec2 r = particles[idx].position - mousePos;
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

        // Clamp force
        float fmag = sqrtf(force.x * force.x + force.y * force.y);
        if (fmag > 1000.0f * particles[idx].density) {
            float s = (1000.0f * particles[idx].density) / fmag;
            force.x *= s;
            force.y *= s;
        }

        // Integrate velocity
        particles[idx].velocity.x += DT * force.x / particles[idx].density;
        particles[idx].velocity.y += DT * force.y / particles[idx].density;

        // Integrate position & boundary (matches serial_integrate)
        particles[idx].position.x += DT * particles[idx].velocity.x;
        particles[idx].position.y += DT * particles[idx].velocity.y;
        float vmag = sqrtf(particles[idx].velocity.x * particles[idx].velocity.x +
            particles[idx].velocity.y * particles[idx].velocity.y);
        if (vmag > MAX_VEL) {
            float s = MAX_VEL / vmag;
            particles[idx].velocity.x *= s;
            particles[idx].velocity.y *= s;
        }
        if (particles[idx].position.x < 0.05f) {
            particles[idx].position.x = 0.05f;
            particles[idx].velocity.x = fabsf(particles[idx].velocity.x) * DAMPING;
        }
        if (particles[idx].position.x > 1.95f) {
            particles[idx].position.x = 1.95f;
            particles[idx].velocity.x = -fabsf(particles[idx].velocity.x) * DAMPING;
        }
        if (particles[idx].position.y < 0.05f) {
            particles[idx].position.y = 0.05f;
            particles[idx].velocity.y = fabsf(particles[idx].velocity.y) * DAMPING;
        }
        if (particles[idx].position.y > 1.95f) {
            particles[idx].position.y = 1.95f;
            particles[idx].velocity.y = -fabsf(particles[idx].velocity.y) * DAMPING;
        }
    }
}

int MPI_main(int N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) cerr << "Error: N must be divisible by number of MPI processes\n";
        MPI_Finalize();
        return -1;
    }
    int N_local = N / size;
    int start = rank * N_local;
    int end = start + N_local;

    // Particle data and communication buffers
    vector<Particle> particles(N);
    vector<Vec2> allPos(N), allVel(N), localPos(N_local), localVel(N_local);
    vector<float> allDens(N), allPres(N), localDens(N_local), localPres(N_local);

    // Initialize on rank 0 and broadcast
    if (rank == 0) mpi_initParticles(particles,N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            allPos[i] = particles[i].position;
            allVel[i] = particles[i].velocity;
        }
    }
    MPI_Bcast(allPos.data(), N * sizeof(Vec2), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(allVel.data(), N * sizeof(Vec2), MPI_BYTE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < N; ++i) {
        particles[i].position = allPos[i];
        particles[i].velocity = allVel[i];
        particles[i].density = REST_DENSITY;
        particles[i].pressure = 0;
        particles[i].valid = true;
    }

    // GLFW setup on rank 0
    GLFWwindow* window = nullptr;
    if (rank == 0) {
        if (!glfwInit()) {
            cerr << "Failed to initialize GLFW\n";
            MPI_Finalize();
            return -1;
        }
        window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "MPI SPH Fluid", NULL, NULL);
        if (!window) {
            glfwTerminate();
            cerr << "Failed to create GLFW window\n";
            MPI_Finalize();
            return -1;
        }
        glfwMakeContextCurrent(window);
        if (glewInit() != GLEW_OK) {
            cerr << "Failed to initialize GLEW\n";
            glfwDestroyWindow(window);
            glfwTerminate();
            MPI_Finalize();
            return -1;
        }
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 2, 0, 2, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glClearColor(0.1f, 0.1f, 0.2f, 1.0f); // Match serial clear color
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
    }

    // FPS variables on rank 0
    double lastTime = glfwGetTime();
    int frameCount = 0;
    double fps = 0.0;
    double totalFrame = 0.0;
    int totalFrameRender = 0;

    // Main simulation loop with synchronized exit
    bool shouldClose = false;
    while (!shouldClose) {
        // Handle mouse input on rank 0 and broadcast
        float mousePosX, mousePosY, interactionStrength;
        if (rank == 0) {
            float simX = (mouseX / WINDOW_WIDTH) * 2.0f;
            float simY = 2.0f - (mouseY / WINDOW_HEIGHT) * 2.0f;
            mousePosX = simX;
            mousePosY = simY;
            interactionStrength = (mouseButtonState == 1) ? 50000.0f : (mouseButtonState == 2) ? -50000.0f : 0.0f;
        }
        else {
            mousePosX = 0.0f;
            mousePosY = 0.0f;
            interactionStrength = 0.0f;
        }
        MPI_Bcast(&mousePosX, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mousePosY, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&interactionStrength, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        Vec2 mousePos(mousePosX, mousePosY);

        // Perform one simulation step
        mpi_computeStep(particles,N, start, end, N_local,
            localPos, localVel, allPos, allVel,
            localDens, localPres, allDens, allPres,
            mousePos, interactionStrength);

        // 4) Gather updated pos & vel for rendering
        for (int i = 0; i < N_local; ++i) {
            localPos[i] = particles[start + i].position;
            localVel[i] = particles[start + i].velocity;
        }
        MPI_Allgather(localPos.data(), N_local * sizeof(Vec2), MPI_BYTE,
            allPos.data(), N_local * sizeof(Vec2), MPI_BYTE, MPI_COMM_WORLD);
        MPI_Allgather(localVel.data(), N_local * sizeof(Vec2), MPI_BYTE,
            allVel.data(), N_local * sizeof(Vec2), MPI_BYTE, MPI_COMM_WORLD);

        // 5) Render and handle FPS on rank 0, synchronize exit
        if (rank == 0) {
            for (int i = 0; i < N; ++i) {
                particles[i].position = allPos[i];
                particles[i].velocity = allVel[i];
            }
            shouldClose = glfwWindowShouldClose(window);
            renderParticles(particles);
            glfwSwapBuffers(window);
            glfwPollEvents();

            // FPS calculation (matches serial_main.cpp)
            double currentTime = glfwGetTime();
            frameCount++;
            if (currentTime - lastTime >= 1.0) {
                fps = frameCount / (currentTime - lastTime);
                totalFrame += fps;
                totalFrameRender++;
                frameCount = 0;
                lastTime = currentTime;
                cout << "FPS: " << fps << endl;
            }
        }
        MPI_Bcast(&shouldClose, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    // Cleanup and average FPS on rank 0
    if (rank == 0) {
        double avgFPS = totalFrame / totalFrameRender;
        cout << "Average FPS: " << avgFPS << " total frame record: " << totalFrameRender << endl;
        glfwDestroyWindow(window);
        glfwTerminate();
    }
	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

float MPI_performance_test(int N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0 && rank == 0) {
        std::cerr << "Error: N must be divisible by number of MPI processes\n";
        MPI_Finalize();
        return 0;
    }
    int N_local = N / size;
    int start = rank * N_local;
    int end = start + N_local;

    std::vector<Particle> particles(N);
    std::vector<Vec2> allPos(N), allVel(N), localPos(N_local), localVel(N_local);
    std::vector<float> allDens(N), allPres(N), localDens(N_local), localPres(N_local);

    if (rank == 0) mpi_initParticles(particles,N);
    MPI_Bcast(particles.data(), N * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);

    int num_steps = 100;
    int num_runs = 10;
    double total_ups = 0.0;

    // Dummy mouse parameters (no interaction in performance test)
    Vec2 mousePos(0.0f, 0.0f);
    float interactionStrength = 0.0f;

    for (int run = 0; run < num_runs; ++run) {
        double start_time = MPI_Wtime();
        if (rank == 0) cout << "Run: " << run + 1 << " / " << num_runs << endl;
        for (int step = 0; step < num_steps; ++step) {
            mpi_computeStep(particles,N, start, end, N_local,
                localPos, localVel, allPos, allVel,
                localDens, localPres, allDens, allPres,
                mousePos, interactionStrength);
            if (rank == 0 )  // Print every 100 steps
                cout << "\nStep: " << step + 1 << " / " << num_steps << endl;
        }
        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        double ups = num_steps / elapsed;
        if (rank == 0) total_ups += ups;
    }

    float avg_ups = 0.0;
    if (rank == 0) {
        avg_ups = total_ups / num_runs;
    }
    return avg_ups;
}