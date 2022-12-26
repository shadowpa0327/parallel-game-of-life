/*
install required packages:
sudo apt-get update
sudo apt-get install freeglut3 freeglut3-dev binutils-gold g++ cmake libglew-dev mesa-common-dev build-essential libglew1.5-dev libglm-dev
run opengl screen:
g++ screen.cpp utils.cpp -o screen -lglut -lGLU -lGL
*/

// Author: Mario Talevski
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <string>
#include <stdlib.h>

#include <chrono>
#include <thread>

#include <GL/glew.h>
#include <GL/freeglut.h>  // GLUT, include glu.h and gl.h

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

// CUDA and GameOfLife correlated header
#include "gameOfLifeCPU.h"
#include "gameOfLifeCUDA.h"
#include "CUDAFunctions.h"
#include "utils.h"
#include "parameter.h"

#if defined(OS_WIN)
  #include <windows.h> // Use for windows
#endif

#define checkCudaErrors(val)    checkCudaResult((val), #val, __FILE__, __LINE__)

template<typename T>
bool checkCudaResult(T result, char const *const func, const char *const file, int const line) {
	if (result) {
		if (result == cudaErrorCudartUnloading) {
			// Do not try to print error when program is shutting down.
			return false;
		}

		std::stringstream ss;
		ss << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
			<< " \"" << func << "\"";
		std::cerr << ss.str() << std::endl;
		return false;
	}
	return true;
}


typedef glm::ivec2 Vector2i;
typedef glm::ivec3 Vector3i;
typedef glm::mediump_vec3 Vector3f;
typedef glm::mediump_vec4 Vector4f;
//typedef glm::detail::tvec4<unsigned char> Vector4c;
typedef unsigned char ubyte;

int cudaDeviceId = -1;

bool menuVisible = false;
int screenWidth = 1920;
int screenHeight = 1080;
bool updateTextureNextFrame = true;
bool resizeTextureNextFrame = true;

Vector2i mousePosition;
int mouseButtons = 0;

bool runGpuLife = true;
bool lifeRunning = false;
bool bitLife = true;
bool useLookupTable = true;
bool useBigChunks = false;
bool parallelCpuLife = false;
bool useCpuParallelLambda = false;
ushort threadsCount = 128;

bool resizeWorld = false;
bool resetWorldPostprocessDisplay = true;

// camera
int zoom = 0;
Vector2i translate = Vector2i(0, 0);
ubyte* textureData = nullptr;
size_t textureWidth = screenWidth;
size_t textureHeight = screenHeight;
bool cyclicWorld = true;
bool useBetterRandom = true;
bool postprocess = true;
bool cpuBench = true;
bool gpuBench = true;
bool individualBench = false;

float lastProcessTime = 0;

// game of life settings
size_t lifeIterations = 1;
uint bitLifeBytesPerTrhead = 1u;

size_t worldWidth = gridWidth;
size_t worldHeight = gridHeight;

size_t newWorldWidth = gridWidth;
size_t newWorldHeight = gridHeight;


uint8_t* d_cpuDisplayData = nullptr;


/// Host-side texture pointer.
uchar4* h_textureBufferData = nullptr;
/// Device-side texture pointer.
uchar4* d_textureBufferData = nullptr;

GLuint gl_pixelBufferObject = 0;
GLuint gl_texturePtr = 0;
cudaGraphicsResource* cudaPboResource = nullptr;


// GameOfLife Object of CPU and GPU
gameOfLifeCPU cpuLife;
gameOfLifeCUDA cudaLife;

// Global buffer for shared data
uint8_t* globalGrid = nullptr;
char initialization_mode; // mode for initialization

void initGlobalGrid(){  
  globalGrid = new uint8_t[worldWidth * worldHeight];
  initGrid(initialization_mode, globalGrid);
}

void freeGlobalBuffers(){
  delete []globalGrid;
  globalGrid = nullptr;
}

void freeLocalBuffers(){
  cpuLife.freeBuffers();
  cudaLife.freeBuffers();
  cudaFree(d_cpuDisplayData);
  d_cpuDisplayData = nullptr;
}

void resizeLifeWorld(size_t newWorldWidth, size_t newWorldHeight){
  freeLocalBuffers();

  worldWidth = newWorldWidth;
  worldHeight = newWorldHeight;

  cpuLife.resizeWorld(worldWidth, worldHeight);
  cudaLife.resizeWorld(worldWidth, worldHeight);
}

void initWorld(bool isCUDA, bool bitlife){
  if(isCUDA){
    cudaLife.initWorld(globalGrid, bitlife);
  }
  else{
    cpuLife.initWorld(globalGrid);
  }
}


float runCpuLife(size_t iterations){
  size_t worldSize = worldHeight * worldWidth;
  if(!cpuLife.areBuffersAllocated()){
    cout << "Initialize the buffer for CPU life\n";
    freeLocalBuffers();
    cpuLife.allocBuffers();
    initWorld(false, false);
    assert(d_cpuDisplayData == nullptr);
    cudaMalloc((void**)&d_cpuDisplayData, worldSize);
  }
  //printGrid(cpuLife.getGridData());
  auto t1 = std::chrono::high_resolution_clock::now();
  cpuLife.iterate(iterations, worldHeight, worldWidth, 1);
  auto t2 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_cpuDisplayData, cpuLife.getGridData(), worldSize, cudaMemcpyHostToDevice);
  //printf("runCopy : %s\n", cudaGetErrorString(cudaGetLastError())); 
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
}

float runCUDALife(size_t iterations, ushort threadsCount, bool bitLife, uint bitLifeBytesPerTrhead) {
  if(!cudaLife.areBuffersAllocated(bitLife)){
    cout << "Initialize the buffer for GPU life\n";
    freeLocalBuffers();
    cudaLife.allocBuffers(bitLife);
    initWorld(true, bitLife);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  cudaLife.iterate(iterations, bitLife, threadsCount, bitLifeBytesPerTrhead);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
}

void runLife(){
  float time;
  if (runGpuLife){
    time = runCUDALife(lifeIterations, threadsCount, bitLife, bitLifeBytesPerTrhead);
  }
  else{
    time = runCpuLife(lifeIterations);
  }
  lastProcessTime = time;
}

void displayLife() {
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaPboResource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource));

	bool ppc = false;
	if(!runGpuLife){
		assert(d_cpuDisplayData != nullptr);
		runDisplayLifeKernel(d_cpuDisplayData, worldWidth, worldHeight, d_textureBufferData, screenWidth, screenHeight, 
							translate.x, translate.y, zoom, ppc, cyclicWorld, false
							);
	}
	else{
		if(!bitLife){
      runDisplayLifeKernel(cudaLife.getGridData(), worldWidth, worldHeight, d_textureBufferData, screenWidth, screenHeight, 
        translate.x, translate.y, zoom, ppc, cyclicWorld, false
      );
    }
    else{
      runDisplayLifeKernel(cudaLife.getEncodedGridData(), worldWidth, worldHeight, d_textureBufferData, screenWidth, screenHeight, 
        translate.x, translate.y, zoom, ppc, cyclicWorld, true
      );
    }
	}
	
	//printf("runDisplayLifeKernel : %s\n", cudaGetErrorString(cudaGetLastError())); 

	
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
}

void drawTexture() {
	glColor3f(1.0f, 1.0f, 1.0f);
	glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);


	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(float(screenWidth), 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(float(screenWidth), float(screenHeight));
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, float(screenHeight));
	glEnd();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}


bool initOpenGlBuffers(int width, int height) {
	// Free any previously allocated buffers

	delete[] h_textureBufferData;
	h_textureBufferData = nullptr;

	glDeleteTextures(1, &gl_texturePtr);
	gl_texturePtr = 0;

	if (gl_pixelBufferObject) {
		cudaGraphicsUnregisterResource(cudaPboResource);
		glDeleteBuffers(1, &gl_pixelBufferObject);
		gl_pixelBufferObject = 0;
	}

	// Check for minimized window or invalid sizes.
	if (width <= 0 || height <= 0) {
		return true;
	}

	// Allocate new buffers.

	h_textureBufferData = new uchar4[width * height];

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_texturePtr);
	glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_textureBufferData);

	glGenBuffers(1, &gl_pixelBufferObject);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), h_textureBufferData, GL_STREAM_COPY);
	// While a PBO is registered to CUDA, it can't be used as the destination for OpenGL drawing calls.
	// But in our particular case OpenGL is only used to display the content of the PBO, specified by CUDA kernels,
	// so we need to register/unregister it only once.
	cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);
	if (result != cudaSuccess) {
		return false;
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	return true;
}

void drawString(float x, float y, float z, const std::string& text) {
	glRasterPos3f(x, y, z);
	for (size_t i = 0; i < text.size(); i++) {
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, text[i]);
	}
}

void drawControls(float dx, float dy) {

		float i = 1;
		float incI = 14;
		std::stringstream ss;

		ss.str("");
		ss << "World size: " << ((newWorldWidth * newWorldHeight) / 1000000.0f) << " millions";
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "Zoom: " << zoom;
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "[i] [o] Life iterations: " << lifeIterations;
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "[+] [-] World width: " << newWorldWidth;
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "[*] [/] World height: " << newWorldHeight;
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "  [r]   Use better random: " << (useBetterRandom ? "yes" : "no");
		drawString(dx, ++i * incI + dy, 0, ss.str());

		++i;

		ss.str("");
		ss << "  [g]   Toggle GPU/CPU: " << (runGpuLife ? "GPU" : "CPU");
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "  [b]   Toggle bit/byte per cell: " << (bitLife ? "bit" : "byte");
		drawString(dx, ++i * incI + dy, 0, ss.str());

		if (bitLife) {
			ss.str("");
			ss << "[n] [m] Bytes per thread: " << bitLifeBytesPerTrhead;
			drawString(dx, ++i * incI + dy, 0, ss.str());

			ss.str("");
			ss << "  [k]   Use lookup table: " << (useLookupTable ? "yes" : "no");
			drawString(dx, ++i * incI + dy, 0, ss.str());

			ss.str("");
			ss << "  [j]   Use big chunks table: " << (useBigChunks ? "yes" : "no");
			drawString(dx, ++i * incI + dy, 0, ss.str());
		}

		if (!runGpuLife && !bitLife) {
			ss.str("");
			ss << "  [p]   Toggle parallel/serial: " << (parallelCpuLife ? "parallel" : "serial");
			drawString(dx, ++i * incI + dy, 0, ss.str());

			if (parallelCpuLife) {
				ss.str("");
				ss << "  [l]   Toggle static call/lambda: " << (useCpuParallelLambda ? "lambda" : "static call");
				drawString(dx, ++i * incI + dy, 0, ss.str());
			}
		}

		++i;

		drawString(dx, ++i * incI + dy, 0, "  [ ]   Run ");

		++i;
		drawString(dx, ++i * incI + dy, 0, "  [y]   Toggle post-process");
		drawString(dx, ++i * incI + dy, 0, "  [a]   Toggle auto-running");
		ss.str("");
		ss << "  [w]   Run CPU in benchmark: " << (cpuBench ? "yes" : "no");
		drawString(dx, ++i * incI + dy, 0, ss.str());
		ss.str("");
		ss << "  [e]   Run individual inters bench: " << (individualBench ? "yes" : "no");
		drawString(dx, ++i * incI + dy, 0, ss.str());
		drawString(dx, ++i * incI + dy, 0, "  [q]   Run benchmark");
		drawString(dx, ++i * incI + dy, 0, "  [h]   Hide/show this menu");


		i = screenHeight / incI - 4;

		ss.str("");
		ss << "Last process time: " << lastProcessTime << " ms";
		drawString(dx, ++i * incI + dy, 0, ss.str());

		ss.str("");
		ss << "Cells per second: " << ((float)(worldWidth * worldHeight) / lastProcessTime) / 1000.0f << " millions";
		drawString(dx, ++i * incI + dy, 0, ss.str());

	}

void displayCallback() {

	if (lifeRunning) {
		runLife();
	}
	displayLife();
	drawTexture();

	if (menuVisible) {
		glColor3f(0.0f, 0.0f, 0.0f);
		drawControls(9, -1);
		drawControls(11, 1);
		glColor3f(0.9f, 0.8f, 0.0f);
		drawControls(10, 0);
	}

	glutSwapBuffers();
	glutReportErrors();

	lifeRunning = true;
}



void reshapeCallback(int w, int h) {
	screenWidth = w;
	screenHeight = h;

	glViewport(0, 0, screenWidth, screenHeight);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, screenWidth, screenHeight, 0.0, -1.0, 1.0);

	initOpenGlBuffers(screenWidth, screenHeight);
	resetWorldPostprocessDisplay = true;
}

void idleCallback() {
	if (lifeRunning){
		//printGrid(gridOne);
		glutPostRedisplay();
	}
	else {
		// Prevent GLUT from eating 100% CPU.
		std::chrono::milliseconds dur(1000 / 30);  // About 30 fps
		std::this_thread::sleep_for(dur);
	}
}



bool initGL(int* argc, char** argv) {
	glutInit(argc, argv);  // Create GL context.
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(screenWidth, screenHeight);
	glutCreateWindow("CPU vs. GPU in Conway's Game of Life by Marek Fiser (http://marekfiser.cz)");

	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0")) {
		std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
		return false;
	}

	glutReportErrors();
	return true;
}


void initCuda() {

	cudaDeviceProp deviceProp;
	cudaChooseDevice(&cudaDeviceId, &deviceProp);
	cudaGetDeviceProperties(&deviceProp, cudaDeviceId);

	int devicesCount = -1;
	cudaGetDeviceCount(&devicesCount);
	printf("There is %d available device(s) on this machine. \n", devicesCount);
	printf("Active device ID: %d\n", cudaDeviceId);

	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	printf("  CUDA Driver Version / Runtime Version:      %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("  CUDA Capability Major/Minor version:        %d.%d\n", deviceProp.major, deviceProp.minor);

	printf("  Total amount of global memory:              %.0f MBytes (%llu bytes)\n",
			(float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long)deviceProp.totalGlobalMem);

	printf("  Multiprocessors count:                      %2d\n", deviceProp.multiProcessorCount);

	printf("  Constant memory:                            %lu bytes\n", deviceProp.totalConstMem);
	printf("  Shared memory per block:                    %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Registers available per block:              %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                  %d\n", deviceProp.warpSize);
	printf("  Maximum threads per multiprocessor:         %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum threads total:                      %d\n",
		deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount);
	printf("  Maximum threads per block:                  %d\n", deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:  %d x %d x %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                       %lu bytes\n", deviceProp.memPitch);
	printf("  Run time limit on kernels:                  %s\n",
		deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
	printf("  Concurrent kernels:                         %d\n", deviceProp.concurrentKernels);

	// If this code causes stack corruption, it is probably cause by function cudaGetDeviceProperties because you
	// compiled the project with incorrect CUDA version. Check that printed version of runtime is the same as
	// version enabled in "Build customization" option of the project.
	// http://stackoverflow.com/questions/11654502/is-cudagetdeviceproperties-returning-corrupted-info
}



void keyboardCallback(unsigned char key, int /*mouseX*/, int /*mouseY*/) {

		switch (key) {
			case 27:  // Esc
				exit(EXIT_SUCCESS);
			case ' ':  // Space - run the life iteration step.
				runLife();
				break;
			case 'o':
				lifeIterations <<= 1;
				break;
			case 'i':
				if (lifeIterations > 1) {
					lifeIterations >>= 1;
				}
				break;
			case 'g':
				runGpuLife = !runGpuLife;
				break;
			case 'b':
				bitLife = !bitLife;
				break;
			case 'n':
				if (bitLifeBytesPerTrhead < 1024) {
					bitLifeBytesPerTrhead <<= 1;
				}
				break;
			case 'm':
				if (bitLifeBytesPerTrhead > 1) {
					bitLifeBytesPerTrhead >>= 1;
				}
				break;
			case 'k':
				useLookupTable = !useLookupTable;
				break;
			case 'j':
				useBigChunks = !useBigChunks;
				break;

			case 'p':
				parallelCpuLife = !parallelCpuLife;
				break;
			case 'l':
				useCpuParallelLambda = !useCpuParallelLambda;
				break;

			case 'y':
				postprocess = !postprocess;
				break;
			case 'c':
				cyclicWorld = !cyclicWorld;
				resetWorldPostprocessDisplay = true;
				break;
			case 'h':
				menuVisible = !menuVisible;
				break;
			case 'a':
				lifeRunning = !lifeRunning;
				break;
		}

		glutPostRedisplay();
	}

void mouseCallback(int button, int state, int x, int y) {

	if (button == 3 || button == 4) { // stroll a wheel event
		// Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
		if (state == GLUT_UP) {
			return; // Disregard redundant GLUT_UP events
		}

		int zoomFactor = (button == 3) ? -1 : 1;
		zoom += zoomFactor;

		resetWorldPostprocessDisplay = true;
		glutPostRedisplay();
		return;
	}

	if (state == GLUT_DOWN) {
		mouseButtons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouseButtons &= ~(1 << button);
	}

	mousePosition.x = x;
	mousePosition.y = y;

	glutPostRedisplay();
}

void motionCallback(int x, int y) {
	int dx = x - mousePosition.x;
	int dy = y - mousePosition.y;

	if (mouseButtons == 1 << GLUT_LEFT_BUTTON) {
		translate.x += dx;
		translate.y += dy;
		resetWorldPostprocessDisplay = true;
	}

	mousePosition.x = x;
	mousePosition.y = y;

	glutPostRedisplay();
}


void runGui(int argc, char** argv){
  initGlobalGrid();
	initGL(&argc, argv);

	glutDisplayFunc(displayCallback);
	glutReshapeFunc(reshapeCallback);
	glutKeyboardFunc(keyboardCallback);
	glutMouseFunc(mouseCallback);
	glutMotionFunc(motionCallback);
	glutIdleFunc(idleCallback);
  	resizeLifeWorld(newWorldWidth, newWorldHeight);
	runLife();

	glutMainLoop();
}



int main(int argc, char** argv) {
	//char mode;
  cout << "Select Initialization mode, read from file or random sample (r/s): ";
  cin >> initialization_mode;
  runGui(argc, argv);	
}