#include "sph.cuh"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "main.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm> 
using namespace std;

// Global device memory
Particle* d_particles = nullptr;
int* d_cellIndices = nullptr;
int* d_cellStart = nullptr;
int* d_cellEnd = nullptr;


//Compute each particle's cell index
__global__ void computeCellIndices(Particle * particles,
        int* cellIndices,
        int   N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float2 pos = particles[i].pos;
    int ix = static_cast<int>(pos.x / SMOOTHING);
    int iy = static_cast<int>(pos.y / SMOOTHING);
    // clamp to valid range
    ix = min(max(ix, 0), GRID_SIZE_X - 1);
    iy = min(max(iy, 0), GRID_SIZE_Y - 1);
    cellIndices[i] = ix + iy * GRID_SIZE_X;
}

//Initialize cellStart/end arrays
__global__ void initCellStartEnd(int* cellStart, int* cellEnd, int  N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    cellStart[c] = N;
    cellEnd[c] = 0;
}\

//Scan sorted indices to find each cell's start and end
__global__ void computeCellStartEnd(int* cellIndices,
    int* cellStart,
    int* cellEnd,
    int  N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = cellIndices[i];
    if (i == 0 || c != cellIndices[i - 1]) {
        atomicMin(&cellStart[c], i);
    }
    if (i == N - 1 || c != cellIndices[i + 1]) {
        atomicMax(&cellEnd[c], i + 1);
    }
}

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

// Compute density and pressure using grid
__global__ void computeDensityPressure(Particle* particles,
    int       N,
    int* cellIndices,
    int* cellStart,
    int* cellEnd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Particle& p = particles[i];
    p.density = 0.0f;
    float2 pos = p.pos;
    int ix = static_cast<int>(pos.x / SMOOTHING);
    int iy = static_cast<int>(pos.y / SMOOTHING);
    // loop over 3×3 neighbor cells
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx < 0 || nx >= GRID_SIZE_X || ny < 0 || ny >= GRID_SIZE_Y) continue;
            int c = nx + ny * GRID_SIZE_X;
            int start = cellStart[c];
            int end = cellEnd[c];
            for (int j = start; j < end; ++j) {
                float2 r; r.x = pos.x - particles[j].pos.x;
                r.y = pos.y - particles[j].pos.y;
                float r2 = r.x * r.x + r.y * r.y;
                if (r2 < SMOOTHING * SMOOTHING) {
                    p.density += MASS * poly6(r2, SMOOTHING);
                }
            }
        }
    }
    float p_term = STIFFNESS * (p.density - REST_DENSITY);
    p.pressure = (p_term > 0.0f ? p_term : 0.0f);
}

// Compute forces using grid
__global__ void computeForces(Particle* particles,
    int       N,
    int* cellIndices,
    int* cellStart,
    int* cellEnd,
    float2    mousePos,
    float     interactionStrength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Particle& p = particles[i];
    if (!p.valid) return;
    float2 pos = p.pos;
    int ix = static_cast<int>(pos.x / SMOOTHING);
    int iy = static_cast<int>(pos.y / SMOOTHING);
    float2 force = { 0.0f, 0.0f };
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx < 0 || nx >= GRID_SIZE_X || ny < 0 || ny >= GRID_SIZE_Y) continue;
            int c = nx + ny * GRID_SIZE_X;
            int start = cellStart[c];
            int end = cellEnd[c];
            for (int j = start; j < end; ++j) {
                if (j == i || !particles[j].valid) continue;
                float2 dr; dr.x = pos.x - particles[j].pos.x;
                dr.y = pos.y - particles[j].pos.y;
                float r2 = dr.x * dr.x + dr.y * dr.y;
                if (r2 >= SMOOTHING * SMOOTHING) continue;
                float rlen = sqrtf(r2);
                float2 grad = spikyGrad(dr, rlen, SMOOTHING);
                float  presTerm = (p.pressure + particles[j].pressure) / (2.0f * particles[j].density);
                float2 pF = { -MASS * presTerm * grad.x,
                              -MASS * presTerm * grad.y };
                float2 rv = { particles[j].vel.x - p.vel.x,
                              particles[j].vel.y - p.vel.y };
                float visc = VISCOSITY * MASS * viscosityLaplacian(rlen, SMOOTHING) / particles[j].density;
                force.x += pF.x + visc * rv.x + ljForce(dr, rlen, SIGMA, EPSILON).x;
                force.y += pF.y + visc * rv.y + ljForce(dr, rlen, SIGMA, EPSILON).y;
            }
        }
    }
    // gravity
    force.y -= 10.0f * p.density;
    // mouse
    if (interactionStrength != 0.0f) {
        float2 d; d.x = pos.x - mousePos.x; d.y = pos.y - mousePos.y;
        float d2 = d.x * d.x + d.y * d.y;
        if (d2 < 0.25f * 0.25f) {
            float dl = sqrtf(d2);
            if (dl > 1e-6f) {
                float mag = interactionStrength * (0.25f - dl) / 0.25f;
                force.x += mag * d.x / dl;
                force.y += mag * d.y / dl;
            }
        }
    }
    // clamp & integrate
    float fmag = sqrtf(force.x * force.x + force.y * force.y);
    if (fmag > 1000.0f * p.density) {
        float s = (1000.0f * p.density) / fmag;
        force.x *= s; force.y *= s;
    }
    p.vel.x += DT * force.x / p.density;
    p.vel.y += DT * force.y / p.density;
}


__global__ void integrate(Particle* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Particle& p = particles[i];
    p.pos.x += DT * p.vel.x;
    p.pos.y += DT * p.vel.y;
    float velMag = sqrtf(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    if (velMag > MAX_VEL) {
        float scale = MAX_VEL / velMag;
        p.vel.x *= scale;
        p.vel.y *= scale;
    }
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

void initSimulation(Particle* particles, int N, cudaGraphicsResource* cudaVBO) {
    if (d_particles == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    }
    if (d_cellIndices == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_cellIndices, N * sizeof(int)));
    }
    if (d_cellStart == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_cellStart, GRID_SIZE * sizeof(int)));
    }
    if (d_cellEnd == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_cellEnd, GRID_SIZE * sizeof(int)));
    }
    Particle* h_particles = new Particle[N];
    int cols = static_cast<int>(std::ceil(std::sqrt(N)));
    int rows = (N + cols - 1) / cols;
    if (rows == 0) rows = 1;
    const float minPos = 0.05f;
    const float maxPos = 1.95f;
    const float range = maxPos - minPos;
    float spacingX = range / (cols > 1 ? cols - 1 : 1);
    float spacingY = range / (rows > 1 ? rows - 1 : 1);
    float spacing = std::min(spacingX, spacingY);
    float startX = minPos + (range - (cols - 1) * spacing) / 2.0f;
    float startY = maxPos - (range - (rows - 1) * spacing) / 2.0f;
    for (int i = 0; i < N; i++) {
        int x = i % cols;
        int y = i / cols;
        h_particles[i].pos = make_float2(startX + x * spacing, startY - y * spacing);
        h_particles[i].vel = make_float2(0.0f, 0.0f);
        h_particles[i].density = REST_DENSITY;
        h_particles[i].pressure = 0.0f;
        h_particles[i].valid = true;
    }
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));
    delete[] h_particles;
}

__global__ void updateVBO(float2* vboPtr, int N, Particle* particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (particles[i].valid) {
        vboPtr[i] = particles[i].pos;
    }
    else {
        vboPtr[i] = make_float2(-1.0f, -1.0f);
    }
}

void stepSimulation(Particle* particles, int N, int* grid, int* cellStart, int* cellEnd, cudaGraphicsResource* cudaVBO, float2 mousePos, float interactionStrength) {
    int threadsPerBlock = 1024;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int gridBlocks = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Compute cell indices
    computeCellIndices << <blocks, threadsPerBlock >> > (d_particles, d_cellIndices, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sort particles by cell indices
    thrust::device_ptr<int>      idx_ptr(d_cellIndices);
    thrust::device_ptr<Particle> p_ptr(d_particles);
    thrust::sort_by_key(idx_ptr, idx_ptr + N, p_ptr);

    // reset start/end
    initCellStartEnd << <gridBlocks, threadsPerBlock >> > (d_cellStart, d_cellEnd, GRID_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    //  compute start/threadsPerBlock
    computeCellStartEnd << <blocks, threadsPerBlock >> > (d_cellIndices, d_cellStart, d_cellEnd, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute density, pressure, and forces using grid
    computeDensityPressure << <blocks, threadsPerBlock >> > (d_particles, N, d_cellIndices, d_cellStart, d_cellEnd);
    computeForces << <blocks, threadsPerBlock >> > (d_particles, N, d_cellIndices, d_cellStart, d_cellEnd, mousePos, interactionStrength);
    integrate << <blocks, threadsPerBlock >> > (d_particles, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update VBO
    if (cudaVBO != nullptr)
    {
        float2* vboPtr;
        size_t size;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBO, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, cudaVBO));
        updateVBO << <blocks, threadsPerBlock >> > (vboPtr, N, d_particles);
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBO, 0));
    }
    
    CUDA_CHECK(cudaMemcpy(particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));
}

float CUDA_performance_test(int N) {
    Particle* d_particles = nullptr;
    int* d_cellIndices = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_cellIndices, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, GRID_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, GRID_SIZE * sizeof(int)));

    Particle* h_particles = new Particle[N];
    initSimulation(h_particles, N, nullptr);
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    delete[] h_particles;


    int threadsPerBlock = 1024;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int gridBlocks = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    int num_steps = 100;
    int num_runs = 10;
    double total_ups = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cout << "Run: " << run + 1 << " / " << num_runs << endl;
        for (int step = 0; step < num_steps; ++step) {
            // One full SPH step (build grid, sort, start/end, density, forces, integrate)
            stepSimulation(d_particles,
                N,
                d_cellIndices,
                d_cellStart,
                d_cellEnd,
                nullptr,
                make_float2(0,0), 0.0 );
            cout << "\nStep: " << step + 1 << " / " << num_steps << endl;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        total_ups += (num_steps / elapsed);
    }

    float avg_ups = static_cast<float>(total_ups / num_runs);

    cudaFree(d_particles);
    cudaFree(d_cellIndices);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);

    return avg_ups;
}

void cleanupSimulation() {
    if (d_particles != nullptr) {
        CUDA_CHECK(cudaFree(d_particles));
        d_particles = nullptr;
    }
    if (d_cellIndices != nullptr) {
        CUDA_CHECK(cudaFree(d_cellIndices));
        d_cellIndices = nullptr;
    }
    if (d_cellStart != nullptr) {
        CUDA_CHECK(cudaFree(d_cellStart));
        d_cellStart = nullptr;
    }
    if (d_cellEnd != nullptr) {
        CUDA_CHECK(cudaFree(d_cellEnd));
        d_cellEnd = nullptr;
    }
}