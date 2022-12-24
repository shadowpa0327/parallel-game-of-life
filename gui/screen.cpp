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
#include "../parameter.h"
#include "../utils.h"
#include <chrono>
#include <thread>
#include "display.h"
//#include "gameOfLife.h"
#include <GL/glew.h>
#include <GL/freeglut.h>  // GLUT, include glu.h and gl.h



#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

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


bool* gridOne = malloc_host<bool>(gridHeight*gridWidth, false);
bool* gridTwo = malloc_host<bool>(gridHeight*gridWidth, false);
bool* gridAns = malloc_host<bool>(gridHeight*gridWidth, false);


int cudaDeviceId = -1;

bool menuVisible = false;
int screenWidth = 1024;
int screenHeight = 1024;
bool updateTextureNextFrame = true;
bool resizeTextureNextFrame = true;

Vector2i mousePosition;
int mouseButtons = 0;

bool runGpuLife = false;
bool lifeRunning = false;
bool bitLife = false;
bool useLookupTable = true;
bool useBigChunks = false;
bool parallelCpuLife = false;
bool useCpuParallelLambda = false;
ushort threadsCount = 256;

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
size_t lifeIteratinos = 1;
uint bitLifeBytesPerTrhead = 1u;

size_t worldWidth = 256;
size_t worldHeight = 256;

size_t newWorldWidth = 256;
size_t newWorldHeight = 256;


ubyte* d_cpuDisplayData = nullptr;


/// Host-side texture pointer.
uchar4* h_textureBufferData = nullptr;
/// Device-side texture pointer.
uchar4* d_textureBufferData = nullptr;

GLuint gl_pixelBufferObject = 0;
GLuint gl_texturePtr = 0;
cudaGraphicsResource* cudaPboResource = nullptr;

void updateSerial(bool* &gridOne, bool* &gridTwo){
    std::swap(gridOne, gridTwo);
    for(int a = 1; a <= gridHeight; a++)
    {
        for(int b = 1; b <= gridWidth; b++)
        {
            int alive =   gridTwo[(a-1)*arrayWidth + b-1]   + gridTwo[a*arrayWidth + b-1] + gridTwo[(a+1)*arrayWidth + b-1]
                        + gridTwo[(a-1)*arrayWidth + b]                                  + gridTwo[(a+1)*arrayWidth + b]
                        + gridTwo[(a-1)*arrayWidth + b+1]   + gridTwo[a*arrayWidth + b+1] + gridTwo[(a+1)*arrayWidth + b+1];;
            gridOne[a*arrayWidth + b] = ((alive == 3) || (alive==2 && gridTwo[a*arrayWidth + b]))?1:0; 
        }
    }
}



void displayLife() {
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaPboResource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource));

	bool ppc = false;


	
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


void displayCallback() {

	if (lifeRunning) {
		//runLife();
	}

	displayLife();
	drawTexture();

	// if (menuVisible) {
	// 	glColor3f(0.0f, 0.0f, 0.0f);
	// 	drawControls(9, -1);
	// 	drawControls(11, 1);
	// 	glColor3f(0.9f, 0.8f, 0.0f);
	// 	drawControls(10, 0);
	// }

	glutSwapBuffers();
	glutReportErrors();
}



void idleCallback() {
	if (lifeRunning){
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

int main(int argc, char** argv) {
	initGL(&argc, argv);
	initCuda();
}

