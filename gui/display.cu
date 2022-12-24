#include <cuda_runtime.h>
#include <assert.h>
#include "../utils.h"
#include "../parameter.h"
#define checkCudaErrors(val)   checkCudaResult((val), #val, __FILE__, __LINE__)

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

/// CUDA kernel for rendering of life world on the screen.
/// This kernel transforms bit-per-cell life world to ARGB screen buffer.
__global__ void displayLifeKernel(const uint8_t* lifeData, uint worldWidth, uint worldHeight, uchar4* destination,
        int destWidth, int detHeight, int2 displacement, double zoomFactor, int multisample, bool simulateColors,
        bool cyclic, bool bitLife) {

    uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;

    int x = (int)floor(((int)(pixelId % destWidth) - displacement.x) * zoomFactor);
    int y = (int)floor(((int)(pixelId / destWidth) - displacement.y) * zoomFactor);

    if (cyclic) {
        x = ((x % (int)worldWidth) + worldWidth) % worldWidth;
        y = ((y % (int)worldHeight) + worldHeight) % worldHeight;
    }
    else if (x < 0 || y < 0 || x >= worldWidth || y >= worldHeight) {
        destination[pixelId].x = 127;
        destination[pixelId].y = 127;
        destination[pixelId].z = 127;
        return;
    }

    int value = 0;  // Start at value - 1.
    int increment = 255 / (multisample * multisample);

    if (bitLife) {
        for (int dy = 0; dy < multisample; ++dy) {
            int yAbs = (y + dy) * worldWidth;
            for (int dx = 0; dx < multisample; ++dx) {
                int xBucket = yAbs + x + dx;
                value += ((lifeData[xBucket >> 3] >> (7 - (xBucket & 0x7))) & 0x1) * increment;
            }
        }
    }
    else {
        for (int dy = 0; dy < multisample; ++dy) {
            int yAbs = (y + dy) * worldWidth;
            for (int dx = 0; dx < multisample; ++dx) {
                value += lifeData[yAbs + (x + dx)] * increment;
            }
        }
    }

    bool isNotOnBoundary = !cyclic || !(x == 0 || y == 0);

    if (simulateColors) {
        if (value > 0) {
            if (destination[pixelId].w > 0) {
                // Stayed alive - get darker.
                if (destination[pixelId].y > 63) {
                    if (isNotOnBoundary) {
                        --destination[pixelId].x;
                    }
                    --destination[pixelId].y;
                    --destination[pixelId].z;
                }
            }
            else {
                // Born - full white color.
                destination[pixelId].x = 255;
                destination[pixelId].y = 255;
                destination[pixelId].z = 255;
            }
        }
        else {
            if (destination[pixelId].w > 0) {
                // Died - dark green.
                if (isNotOnBoundary) {
                    destination[pixelId].x = 0;
                }
                destination[pixelId].y = 128;
                destination[pixelId].z = 0;
            }
            else {
                // Stayed dead - get darker.
                if (destination[pixelId].y > 8) {
                    if (isNotOnBoundary) {
                    }
                    destination[pixelId].y -= 8;
                }
            }
        }
    }
    else {
        destination[pixelId].x = isNotOnBoundary ? value : 255;
        destination[pixelId].y = value;
        destination[pixelId].z = value;
    }

    // Save last state of the cell to the alpha channel that is not used in rendering.
    destination[pixelId].w = value;
}

/// Runs a kernel for rendering of life world on the screen.
void runDisplayLifeKernel(const uint8_t* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
        int destWidth, int destHeight, int displacementX, int displacementY, int zoom, bool simulateColors,
        bool cyclic, bool bitLife) {

    ushort threadsCount = 256;
    assert((worldWidth * worldHeight) % threadsCount == 0);
    size_t reqBlocksCount = (destWidth * destHeight) / threadsCount;
    assert(reqBlocksCount < 65536);
    ushort blocksCount = (ushort)reqBlocksCount;

    int multisample = std::min(4, (int)std::pow(2, std::max(0, zoom)));
    displayLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, uint(worldWidth), uint(worldHeight), destination,
        destWidth, destHeight, make_int2(displacementX, displacementY), std::pow(2, zoom),
        multisample, zoom > 1 ? false : simulateColors, cyclic, bitLife);
    checkCudaErrors(cudaDeviceSynchronize());
}