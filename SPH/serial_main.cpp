#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "sph.h"
#include "main.h"

using namespace std;

static double mouseX, mouseY; static int mouseButtonState = 0;

static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	mouseX = xpos; mouseY = ypos;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
		mouseButtonState = 1;
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
		mouseButtonState = 2;
	else mouseButtonState = 0;
}

int serial_main(int N) {
	glfwSetErrorCallback(errorCallback);
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	GLFWwindow* window = glfwCreateWindow(800, 800, "SPH Simulation serial", NULL, NULL);
	if (!window) {
		glfwTerminate();
		std::cerr << "Failed to create GLFW window" << std::endl;
		return -1;
	}
	glfwMakeContextCurrent(window);
	glViewport(0, 0, 800, 800);

	glfwSetCursorPosCallback(window, cursorPositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	vector<Particle> particles;
	serial_initSimulation(particles,N);

	glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 2, 0, 2, -1, 1);
	glMatrixMode(GL_MODELVIEW);

	double lastTime = glfwGetTime();
	int frameCount = 0;
	double fps = 0.0;
	int totalFrame = 0;
	int totalFrameRender = 0;

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);

		float simX = (mouseX / 800.0) * 2.0;
		float simY = 2.0 - (mouseY / 800.0) * 2.0;
		float2 mousePos = { simX, simY };
		float interactionStrength = (mouseButtonState == 1) ? 50000.0f : (mouseButtonState == 2) ? -50000.0f : 0.0f;

		serial_stepSimulation(particles,N, mousePos, interactionStrength);

		double currentTime = glfwGetTime();
		frameCount++;
		if (currentTime - lastTime >= 1.0) {
			fps = frameCount / (currentTime - lastTime);
			totalFrame += fps;
			totalFrameRender++;
			frameCount = 0;
			lastTime = currentTime;
		}

		glEnable(GL_POINT_SMOOTH);
		glPointSize(PARTICLE_SIZE);
		glBegin(GL_POINTS);
		glColor3f(0.0f, 0.5f, 1.0f);
		for (size_t i = 0; i < particles.size(); ++i) {
			const auto& p = particles[i];
			if (p.valid && !std::isnan(p.pos.x) && !std::isnan(p.pos.y)) {
				glVertex2f(p.pos.x, p.pos.y);
			}
			
		}
		glEnd();

		glColor3f(1.0f, 1.0f, 1.0f);
		char fpsText[16];
		snprintf(fpsText, sizeof(fpsText), "FPS: %.0f", fps);
		renderString(0.05f, 1.9f, fpsText, 0.01f);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	double avgFPS = static_cast<double>(totalFrame) / totalFrameRender;
	std::cout << "Average FPS: " << avgFPS << " total frame recode : " << totalFrameRender << std::endl;

	return 0;
}


float serial_performance_test(int N) {
	std::vector<Particle> particles;
	serial_initSimulation(particles,N);  // Initialize particles

	int num_steps = 100;  // Number of simulation steps per run
	int num_runs = 10;     // Number of runs to average over
	double total_ups = 0.0;

	// Set mouse position and interaction strength to zero (no external interaction)
	float2 mousePos = { 0.0f, 0.0f };
	float interactionStrength = 0.0f;

	for (int run = 0; run < num_runs; ++run) {
		serial_initSimulation(particles, N);
		auto start_time = std::chrono::high_resolution_clock::now();
		cout << "Run: " << run + 1 << " / " << num_runs << endl;
		for (int step = 0; step < num_steps; ++step) {
			serial_stepSimulation(particles,N, mousePos, interactionStrength);
			cout << "\nStep: " << step +1 << " / " << num_steps << endl;
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end_time - start_time;
		double ups = num_steps / elapsed.count();
		total_ups += ups;
	
	}

	float avg_ups = total_ups / num_runs;
	return avg_ups;
}