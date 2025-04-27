#include "sph.h"
#include <iostream>
using namespace std;

// Grid type definition
using Grid = std::unordered_map<int, std::vector<int>>;

// Helper function to compute cell index from position
int serial_getCellIndex(float x, float y, float h) {
	int ix = static_cast<int>(x / h);
	int iy = static_cast<int>(y / h);
	return ix + iy * static_cast<int>(2.0 / h + 1);
}

void serial_buildGrid(const std::vector<Particle>& particles, Grid& grid, float h) {
	grid.clear();
	for (int i = 0; i < particles.size(); ++i) {
		const auto& p = particles[i];
		int cellIndex = serial_getCellIndex(p.pos.x, p.pos.y, h);
		grid[cellIndex].push_back(i);
	}
}

// Poly6 kernel
float serial_poly6(float r2, float h) {
	float h2 = h * h;
	if (r2 > h2) return 0.0f;
	float term = h2 - r2;
	return (315.0f / (64.0f * 3.14159f * powf(h, 9))) * term * term * term;
}

// Spiky kernel gradient (softened)
float2 serial_spikyGrad(float2 r, float r_len, float h) {
	if (r_len > h || r_len < 1e-6) return float2(0, 0);
	float term = h - r_len;
	float coeff = -45.0f / (3.14159f * powf(h, 6)) * term * term / r_len;
	return float2(coeff * r.x, coeff * r.y);
}

// Viscosity kernel Laplacian
float serial_viscosityLaplacian(float r, float h) {
	if (r > h) return 0.0f;
	return 45.0f / (3.14159f * powf(h, 6)) * (h - r);
}

// Lennard-Jones force
float2 serial_ljForce(float2 r, float r_len, float sigma, float epsilon) {
	if (r_len >= 2.5f * sigma || r_len < 1e-6f) return float2(0, 0);
	float inv_r = 1.0f / r_len;
	float sigma_over_r = sigma * inv_r;
	float sigma_over_r2 = sigma_over_r * sigma_over_r;
	float sigma_over_r6 = sigma_over_r2 * sigma_over_r2 * sigma_over_r2;
	float sigma_over_r12 = sigma_over_r6 * sigma_over_r6;
	float inv_r2 = inv_r * inv_r;
	float coeff = 24.0f * epsilon * (2.0f * sigma_over_r12 - sigma_over_r6) * inv_r2;
	return float2(coeff * r.x, coeff * r.y);
}

void serial_computeDensityPressure(std::vector<Particle>& particles, int N) {
	Grid grid;
	serial_buildGrid(particles, grid, H);
	for (int i = 0; i < N; i++) {
		Particle& p = particles[i];
		p.density = 0.0f;
		int ix = static_cast<int>(p.pos.x / H);
		int iy = static_cast<int>(p.pos.y / H);
		// Check the particle's cell and its 8 neighbors
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				int neighborX = ix + dx;
				int neighborY = iy + dy;
				int cellIndex = serial_getCellIndex(neighborX * H, neighborY * H, H);
				if (grid.find(cellIndex) != grid.end()) {
					for (int j : grid[cellIndex]) {
						float2 r = { p.pos.x - particles[j].pos.x, p.pos.y - particles[j].pos.y };
						float r2 = r.x * r.x + r.y * r.y;
						if (r2 < H * H) {
							p.density += MASS * serial_poly6(r2, H);
						}
					}
				}
			}
		}
		p.pressure = STIFFNESS * (p.density - REST_DENSITY);
		if (p.pressure < 0.0f) p.pressure = 0.0f;
	}
}

void serial_computeForces(std::vector<Particle>& particles, int N, float2 mousePos, float interactionStrength) {
	Grid grid;
	serial_buildGrid(particles, grid, H);
	for (int i = 0; i < N; i++) {
		if (!particles[i].valid) continue;
		Particle& p = particles[i];
		float2 force = { 0.0f, 0.0f };
		int ix = static_cast<int>(p.pos.x / H);
		int iy = static_cast<int>(p.pos.y / H);
		// Check the particle's cell and its 8 neighbors
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				int neighborX = ix + dx;
				int neighborY = iy + dy;
				int cellIndex = serial_getCellIndex(neighborX * H, neighborY * H, H);
				if (grid.find(cellIndex) != grid.end()) {
					for (int j : grid[cellIndex]) {
						if (j == i || !particles[j].valid) continue;
						float2 r = { p.pos.x - particles[j].pos.x, p.pos.y - particles[j].pos.y };
						float r2 = r.x * r.x + r.y * r.y;
						if (r2 >= H * H) continue;
						float r_len = sqrtf(r2);
						float2 grad = serial_spikyGrad(r, r_len, H);
						float pressureTerm = (p.pressure + particles[j].pressure) / (2.0f * particles[j].density);
						float2 pressureForce = { -MASS * pressureTerm * grad.x, -MASS * pressureTerm * grad.y };
						float2 relVel = { particles[j].vel.x - p.vel.x, particles[j].vel.y - p.vel.y };
						float viscForce = VISCOSITY * MASS * serial_viscosityLaplacian(r_len, H) / particles[j].density;
						float2 ljF = serial_ljForce(r, r_len, SIGMA, EPSILON);
						force.x += pressureForce.x + viscForce * relVel.x + ljF.x;
						force.y += pressureForce.y + viscForce * relVel.y + ljF.y;
					}
				}
			}
		}
		force.y -= 10.0f * p.density;
		if (interactionStrength != 0.0f) {
			float2 r = { p.pos.x - mousePos.x, p.pos.y - mousePos.y };
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

void serial_integrate(std::vector<Particle>& particles, int N) {
	for (int i = 0; i < N; i++) {
		if (!particles[i].valid) continue;
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
}

void serial_initSimulation(std::vector<Particle>& particles, int N) {
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
	float spacing = std::min(spacingX, spacingY); // Use smaller spacing to avoid overlap

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

void serial_stepSimulation(std::vector<Particle>& particles, int N, float2 mousePos, float interactionStrength) {
	serial_computeDensityPressure(particles,N);
	serial_computeForces(particles,N, mousePos, interactionStrength);
	serial_integrate(particles, N);
}