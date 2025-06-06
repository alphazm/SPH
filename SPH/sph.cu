#include "sph.cuh"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "main.h"
using namespace std;

// Global device memory
Particle* d_particles = nullptr;


// Poly6 kernel
__device__ float poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 > h2) return 0.0f;
    float term = h2 - r2;
    return (315.0f / (64.0f * 3.14159f * powf(h, 9))) * term * term * term;
}

// Spiky kernel gradient (softened)
__device__ float2 spikyGrad(float2 r, float r_len, float h) {
    if (r_len > h || r_len < 1e-6) return make_float2(0, 0);
    float term = h - r_len;
    float coeff = -45.0f / (3.14159f * powf(h, 6)) * term * term / r_len;
    return make_float2(coeff * r.x, coeff * r.y);
}

// Viscosity kernel Laplacian
__device__ float viscosityLaplacian(float r, float h) {
    if (r > h) return 0.0f;
    return 45.0f / (3.14159f * powf(h, 6)) * (h - r);
}

// Lennard-Jones force
__device__ float2 ljForce(float2 r, float r_len, float sigma, float epsilon) {
    if (r_len >= 2.5f * sigma || r_len < 1e-6f) return make_float2(0, 0);
    float inv_r = 1.0f / r_len;
    float sigma_over_r = sigma * inv_r;
    float sigma_over_r2 = sigma_over_r * sigma_over_r;
    float sigma_over_r6 = sigma_over_r2 * sigma_over_r2 * sigma_over_r2;
    float sigma_over_r12 = sigma_over_r6 * sigma_over_r6;
    float inv_r2 = inv_r * inv_r;
    float coeff = 24.0f * epsilon * (2.0f * sigma_over_r12 - sigma_over_r6) * inv_r2;
    return make_float2(coeff * r.x, coeff * r.y);
}

__global__ void computeDensityPressure(Particle* particles,int N, int* grid, int* cellStart, int* cellEnd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Particle& p = particles[i];
    p.density = 0.0f;

    for (int j = 0; j < N; j++) {
        float2 r;
        r.x = p.pos.x - particles[j].pos.x;
        r.y = p.pos.y - particles[j].pos.y;
        float r2 = r.x * r.x + r.y * r.y;
        if (r2 < H * H) {
            p.density += MASS * poly6(r2, H);
        }
    }
    p.pressure = STIFFNESS * (p.density - REST_DENSITY);
    if (p.pressure < 0.0f) p.pressure = 0.0f;
}

__global__ void computeForces(Particle* particles,int N, int* grid, int* cellStart, int* cellEnd, float2 mousePos, float interactionStrength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !particles[i].valid) return;

    Particle& p = particles[i];
    float2 force = make_float2(0, 0);

    for (int j = 0; j < N; j++) {
        if (j == i || !particles[j].valid) continue;
        float2 r;
        r.x = p.pos.x - particles[j].pos.x;
        r.y = p.pos.y - particles[j].pos.y;
        float r2 = r.x * r.x + r.y * r.y;
        if (r2 >= H * H) continue;
        float r_len = sqrtf(r2);

        float2 grad = spikyGrad(r, r_len, H);
        float pressureTerm = (p.pressure + particles[j].pressure) / (2.0f * particles[j].density);
        float2 pressureForce = make_float2(-MASS * pressureTerm * grad.x, -MASS * pressureTerm * grad.y);

        float2 relVel;
        relVel.x = particles[j].vel.x - p.vel.x;
        relVel.y = particles[j].vel.y - p.vel.y;
        float viscForce = VISCOSITY * MASS * viscosityLaplacian(r_len, H) / particles[j].density;

        float2 ljF = ljForce(r, r_len, SIGMA, EPSILON);

        force.x += pressureForce.x + viscForce * relVel.x + ljF.x;
        force.y += pressureForce.y + viscForce * relVel.y + ljF.y;
    }
    force.y -= 10.0f * p.density;

    float2 mouseForce = make_float2(0, 0);
    if (interactionStrength != 0.0f) {
        float2 r;
        r.x = p.pos.x - mousePos.x;
        r.y = p.pos.y - mousePos.y;
        float r2 = r.x * r.x + r.y * r.y;
        if (r2 < 0.25f * 0.25f) {
            float r_len = sqrtf(r2);
            if (r_len > 1e-6f) {
                float forceMag = interactionStrength * (0.25f - r_len) / 0.25f;
                mouseForce.x = forceMag * r.x / r_len;
                mouseForce.y = forceMag * r.y / r_len;
                force.x += mouseForce.x;
                force.y += mouseForce.y;
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

__global__ void integrate(Particle* particles,int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Particle& p = particles[i];
    p.pos.x += DT * p.vel.x;
    p.pos.y += DT * p.vel.y;

    // Cap velocity
    float velMag = sqrtf(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    if (velMag > MAX_VEL) {
        float scale = MAX_VEL / velMag;
        p.vel.x *= scale;
        p.vel.y *= scale;
    }

    // Boundary conditions
    if (p.pos.x < 0.05f) {
        p.pos.x = 0.05f;
        p.vel.x = fabsf(p.vel.x) * DAMPING;
    }
    if (p.pos.x > 1.95f) {
        p.pos.x = 1.95f;
        p.vel.x = -fabsf(p.vel.x) * DAMPING;
    }
    if (p.pos.y < 0.05f) {
        p.pos.y = 0.05f;
        p.vel.y = fabsf(p.vel.y) * DAMPING;
    }
    if (p.pos.y > 1.95f) {
        p.pos.y = 1.95f;
        p.vel.y = -fabsf(p.vel.y) * DAMPING;
    }
}

void initSimulation(Particle* particles,int N, cudaGraphicsResource* cudaVBO) {
    if (d_particles == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    }
    Particle* h_particles = new Particle[N];

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
    float spacing = std::min(spacingX, spacingY); // Use smaller spacing to avoid overlap

    // Center the grid
    float startX = minPos + (range - (cols - 1) * spacing) / 2.0f;
    float startY = maxPos - (range - (rows - 1) * spacing) / 2.0f;

    // Initialize particles on host
    for (int i = 0; i < N; i++) {
        int x = i % cols; // Column index
        int y = i / cols; // Row index
        h_particles[i].pos = make_float2(startX + x * spacing, startY - y * spacing);
        h_particles[i].vel = make_float2(0.0f, 0.0f);
        h_particles[i].density = REST_DENSITY;
        h_particles[i].pressure = 0.0f;
        h_particles[i].valid = true;
    }
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));
    // Copy back to host (if needed for VBO or other purposes)
    CUDA_CHECK(cudaMemcpy(particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));
}

__global__ void updateVBO(float2* vboPtr,int N, Particle* particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles[i].valid) {
        vboPtr[i] = particles[i].pos;
    }
    else {
        vboPtr[i] = make_float2(-1.0f, -1.0f);
    }
}

void stepSimulation(Particle* particles,int N, int* grid, int* cellStart, int* cellEnd, cudaGraphicsResource* cudaVBO, float2 mousePos, float interactionStrength) {
    int threadsPerBlock = 1024;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    computeDensityPressure << <blocks, threadsPerBlock >> > (d_particles, N, nullptr, nullptr, nullptr);
    computeForces << <blocks, threadsPerBlock >> > (d_particles, N, nullptr, nullptr, nullptr, mousePos, interactionStrength);
    integrate << <blocks, threadsPerBlock >> > (d_particles, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update VBO only for valid particles
    float2* vboPtr;
    size_t size;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBO, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, cudaVBO));
    updateVBO << <blocks, threadsPerBlock >> > (vboPtr,N, d_particles);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBO, 0));

    CUDA_CHECK(cudaMemcpy(particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));
}

float CUDA_performance_test(int N) {
    Particle* d_particles;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    Particle* h_particles = new Particle[N];
    initSimulation(h_particles,N, nullptr); // Initialize without VBO
    cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice);

    int num_steps = 100;
    int num_runs = 10;
    double total_ups = 0.0;
    int threadsPerBlock = 1024;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int run = 0; run < num_runs; ++run) {
        auto start_time = std::chrono::high_resolution_clock::now();
		cout << "Run " << run + 1 << " of " << num_runs << endl;
        for (int step = 0; step < num_steps; ++step) {
            computeDensityPressure << <blocks, threadsPerBlock >> > (d_particles, N, nullptr, nullptr, nullptr);
            computeForces << <blocks, threadsPerBlock >> > (d_particles, N, nullptr, nullptr, nullptr, make_float2(0, 0), 0.0f);
            integrate << <blocks, threadsPerBlock >> > (d_particles, N);
            cudaDeviceSynchronize();
			cout << "\nStep " << step + 1 << " of " << num_steps << endl;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double ups = num_steps / elapsed.count();
        total_ups += ups;
    }

    float avg_ups = total_ups / num_runs;
    

    cudaFree(d_particles);
    delete[] h_particles;
    return avg_ups;
}

void cleanupSimulation() {
    if (d_particles != nullptr) {
        CUDA_CHECK(cudaFree(d_particles));
        d_particles = nullptr;
    }
}