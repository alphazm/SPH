#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
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

int serial_main() {
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

	std::vector<Particle> particles;
	serial_initSimulation(particles);

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

		serial_stepSimulation(particles, mousePos, interactionStrength);

		double currentTime = glfwGetTime();
		frameCount++;
		if (currentTime - lastTime >= 1.0) {
			fps = frameCount / (currentTime - lastTime);
			totalFrame += fps;
			totalFrameRender++;
			frameCount = 0;
			lastTime = currentTime;
		}

		glPointSize(PARTICLE_SIZE);
		glBegin(GL_POINTS);
		glColor3f(0.0f, 0.5f, 1.0f);
		for (const auto& p : particles) {
			if (p.valid) {
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
	std::cout << "Average FPS: " << avgFPS << " total frame render : " << totalFrameRender << std::endl;

	return 0;
}