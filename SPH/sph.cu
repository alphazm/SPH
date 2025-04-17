#include "sph.cuh"
#include <math.h>
#include <stdio.h>



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

__global__ void computeDensityPressure(Particle* particles, int* grid, int* cellStart, int* cellEnd) {
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

__global__ void computeForces(Particle* particles, int* grid, int* cellStart, int* cellEnd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Particle& p = particles[i];
    float2 force = make_float2(0, 0);

    for (int j = 0; j < N; j++) {
        if (j == i)
            continue;
        float2 r;
        r.x = p.pos.x - particles[j].pos.x;
        r.y = p.pos.y - particles[j].pos.y;
        float r2 = r.x * r.x + r.y * r.y;
        if (r2 >= H * H) continue;
        float r_len = sqrtf(r2);

        // Pressure force
        float2 grad = spikyGrad(r, r_len, H);
        float pressureTerm = (p.pressure + particles[j].pressure) / (2.0f * particles[j].density);
        float2 pressureForce = make_float2(-MASS * pressureTerm * grad.x, -MASS * pressureTerm * grad.y);

        // Viscosity force
        float2 relVel;
        relVel.x = particles[j].vel.x - p.vel.x;
        relVel.y = particles[j].vel.y - p.vel.y;
        float viscForce = VISCOSITY * MASS * viscosityLaplacian(r_len, H) / particles[j].density;

        // Lennard-Jones force
        float2 ljF = ljForce(r, r_len, SIGMA, EPSILON);

        // Accumulate forces
        force.x += pressureForce.x + viscForce * relVel.x + ljF.x;
        force.y += pressureForce.y + viscForce * relVel.y + ljF.y;
    }
    // Gravity
    force.y -= 9.81f * p.density;

    // Cap force magnitude
    float forceMag = sqrtf(force.x * force.x + force.y * force.y);
    if (forceMag > 1000.0f * p.density) {
        float scale = 1000.0f * p.density / forceMag;
        force.x *= scale;
        force.y *= scale;
    }

    // Update velocity
    p.vel.x += DT * force.x / p.density;
    p.vel.y += DT * force.y / p.density;
}

__global__ void integrate(Particle* particles) {
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

void initSimulation(Particle* particles, cudaGraphicsResource* cudaVBO) {
    Particle* d_particles;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    Particle h_particles[N];
    for (int i = 0; i < N; i++) {
        h_particles[i].pos = make_float2(0.7f + 0.05f * (i % 20), 1.5f - 0.05f * (i / 20));
        h_particles[i].vel = make_float2(0, 0);
        h_particles[i].density = REST_DENSITY;
        h_particles[i].pressure = 0;
    }
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_particles));
}

void stepSimulation(Particle* particles, int* grid, int* cellStart, int* cellEnd, cudaGraphicsResource* cudaVBO) {
    Particle* d_particles;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(d_particles, particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeDensityPressure << <blocks, threadsPerBlock >> > (d_particles, nullptr, nullptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    computeForces << <blocks, threadsPerBlock >> > (d_particles, nullptr, nullptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    integrate << <blocks, threadsPerBlock >> > (d_particles);
    CUDA_CHECK(cudaDeviceSynchronize());

    float2* vboPtr;
    size_t size;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBO, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, cudaVBO));
    CUDA_CHECK(cudaMemcpy(vboPtr, d_particles, N * sizeof(float2), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBO, 0));

    CUDA_CHECK(cudaMemcpy(particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_particles));
}