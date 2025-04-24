#ifndef SPH_CUH
#define SPH_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1024        // Number of particles
#define H 0.15f      // Adjusted smoothing radius for sufficient neighbors
#define DT 0.0005f   // Keep small time step
#define MASS 0.05f   // Reduced mass
#define VISCOSITY 0.3f // Slightly higher for damping
#define STIFFNESS 1000.0f // Reduced for stability
#define REST_DENSITY 1000.0f
#define DAMPING 0.9f  // Boundary damping
#define MAX_VEL 5.0f  // Velocity cap
#define SIGMA 0.05f   // Lennard-Jones sigma
#define EPSILON 1.0f  // Lennard-Jones epsilon
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (code %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(1); \
    } \
}

struct Particle {
    float2 pos;
    float2 vel;
    float density;
    float pressure;
    bool valid;
};

__global__ void computeDensityPressure(Particle* particles, int* grid, int* cellStart, int* cellEnd);
__global__ void computeForces(Particle* particles, int* grid, int* cellStart, int* cellEnd, float2 mousePos, float interactionStrength);
__global__ void integrate(Particle* particles);
void initSimulation(Particle* particles, cudaGraphicsResource* cudaVBO);
void stepSimulation(Particle* particles, int* grid, int* cellStart, int* cellEnd, cudaGraphicsResource* cudaVBO, float2 mousePos, float interactionStrength);
void cleanupSimulation();

#endif