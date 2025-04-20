#pragma once
#ifndef SPH_H
#define SPH_H

#include <vector>
#include <cmath>

#define N 500        // Number of particles
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

struct float2 {
	float x, y;
	float2(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
};

struct Particle {
	float2 pos;
	float2 vel;
	float density;
	float pressure;
	bool valid;
};

void serial_computeDensityPressure(std::vector<Particle>& particles);
void serial_computeForces(std::vector<Particle>& particles, float2 mousePos, float interactionStrength);
void serial_integrate(std::vector<Particle>& particles);
void serial_initSimulation(std::vector<Particle>& particles);
void serial_stepSimulation(std::vector<Particle>& particles, float2 mousePos, float interactionStrength);

#endif