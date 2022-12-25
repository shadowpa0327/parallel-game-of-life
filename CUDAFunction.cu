#include <cuda.h>
#include "CUDAFunctions.h"

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


__global__ void updateCUDAKernel(bool* gridOne, bool* gridTwo){
    uint worldSize = gridWidth * gridHeight;
    for(uint cellID = blockIdx.x * blockDim.x + threadIdx.x;
        cellID < worldSize;
        cellID += blockDim.x * gridDim.x){
        uint x = (cellID % (gridWidth)); // 1 base
        uint y = cellID - x;  // 1 base
        uint xLeft = (x + gridWidth - 1)%gridWidth;
        uint xRight = (x + 1) % gridWidth;
        uint yUp = (y + worldSize - gridWidth) % worldSize;
        uint yDown = (y + gridWidth) % worldSize;

        uint alive =      gridTwo[xLeft + yUp]   + gridTwo[x + yUp]   + gridTwo[xRight + yUp] +
                          gridTwo[xLeft + y]     +                    + gridTwo[xRight + y] +
                          gridTwo[xLeft + yDown] + gridTwo[x + yDown] + gridTwo[xRight + yDown];
        
        gridOne[x + y] = alive == 3 || (alive == 2 && gridTwo[x + y]) ? 1 : 0 ;   
    }
}

bool runSimpleLifeKernel(uint8_t*& d_lifeData, uint8_t*& d_lifeDataBuffer, size_t worldWidth, size_t worldHeight,
			size_t iterationsCount, ushort threadsCount){
    
    if((worldWidth * worldHeight) % threadCount != 0){
        return false;
    }

    size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;

    for (size_t i = 0; i < iterationsCount; i++){
        std::swap(d_lifeData, d_lifeDataBuffer);
        updateCUDAKernel<<<reqBlocksCount, threadCount>>>(d_gridOne, d_gridTwo);
    }    
    return true;
}

__global__ void bitLifeEncodeKernel(const uint8_t* lifeData, size_t encWorldSize, uint8_t* resultEncodedLifeData) {
    for (size_t outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
            outputBucketId < encWorldSize;
            outputBucketId += blockDim.x * gridDim.x) {

        size_t cellId = outputBucketId << 3;

        uint8_t result = lifeData[cellId] << 7 | lifeData[cellId + 1] << 6 | lifeData[cellId + 2] << 5
            | lifeData[cellId + 3] << 4 | lifeData[cellId + 4] << 3 | lifeData[cellId + 5] << 2
            | lifeData[cellId + 6] << 1 | lifeData[cellId + 7];

        resultEncodedLifeData[outputBucketId] = result;
    }
    
}

bool runBitEncodedKernel(uint8_t*& gridOne, uint8_t*& gridTwo, size_t gridWidth, size_t gridHeight){
    assert(worldWidth % 8 == 0);
    size_t worldEncDataWidth = worldWidth / 8;
    size_t encWorldSize = worldEncDataWidth * worldHeight;

    ushort threadsCount = min(encWorldSize, (size_t)128);
    assert(encWorldSize % threadsCount == 0);
    size_t reqBlocksCount = encWorldSize / threadsCount;
    ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

    bitLifeEncodeKernel<<<blocksCount, threadsCount>>>(d_lifeData, encWorldSize, d_encodedLife);
    cudaDeviceSynchronize();
}


__global__ void bitLifeDecodeKernel(const uint8_t* encodedLifeData, uint encWorldSize, uint8_t* resultDecodedlifeData) {

    for (uint outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
            outputBucketId < encWorldSize;
            outputBucketId += blockDim.x * gridDim.x) {

        uint cellId = outputBucketId << 3;
        uint8_t dataBucket = encodedLifeData[outputBucketId];

        resultDecodedlifeData[cellId] = dataBucket >> 7;
        resultDecodedlifeData[cellId + 1] = (dataBucket >> 6) & 0x01;
        resultDecodedlifeData[cellId + 2] = (dataBucket >> 5) & 0x01;
        resultDecodedlifeData[cellId + 3] = (dataBucket >> 4) & 0x01;
        resultDecodedlifeData[cellId + 4] = (dataBucket >> 3) & 0x01;
        resultDecodedlifeData[cellId + 5] = (dataBucket >> 2) & 0x01;
        resultDecodedlifeData[cellId + 6] = (dataBucket >> 1) & 0x01;
        resultDecodedlifeData[cellId + 7] = dataBucket & 0x01;
    }

}


/// Runs a kernel that decodes data from bit-per-cell to byte-per-cell format.
void runBitLifeDecodeKernel(const uint8_t* d_encodedLife, uint worldWidth, uint worldHeight, uint8_t* d_lifeData) {

    assert(worldWidth % 8 == 0);
    int worldEncDataWidth = worldWidth / 8;
    int encWorldSize = worldEncDataWidth * worldHeight;

    int threadsCount = min(128, encWorldSize);
    assert(encWorldSize % threadsCount == 0);
    int reqBlocksCount = encWorldSize / threadsCount;
    int blocksCount = std::min(32768, reqBlocksCount);

    // decode life data back to byte per cell format
    bitLifeDecodeKernel<<<blocksCount, threadsCount>>>(d_encodedLife, encWorldSize, d_lifeData);
    cudaDeviceSynchronize();
}


__global__ void bitLifeKernelCounting(const uint8_t* lifeData, uint worldDataWidth, uint worldHeight,
			uint bytesPerThread, uint8_t* resultLifeData) {

    uint worldSize = (worldDataWidth * worldHeight);

    for (uint cellId = (blockIdx.x * blockDim.x + threadIdx.x) * bytesPerThread;
            cellId < worldSize;
            cellId += blockDim.x * gridDim.x * bytesPerThread) {


        uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
        uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
        uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
        uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

        // Initialize data with previous byte and current byte.
        uint data0 = (uint)lifeData[x + yAbsUp] << 16;
        uint data1 = (uint)lifeData[x + yAbs] << 16;
        uint data2 = (uint)lifeData[x + yAbsDown] << 16;

        x = (x + 1) % worldDataWidth;
        data0 |= (uint)lifeData[x + yAbsUp] << 8;
        data1 |= (uint)lifeData[x + yAbs] << 8;
        data2 |= (uint)lifeData[x + yAbsDown] << 8;
        #pragma unroll 8
        for (uint i = 0; i < bytesPerThread; ++i) {
            uint oldX = x;  // Old x is referring to current center cell.
            x = (x + 1) % worldDataWidth;
            data0 |= (uint)lifeData[x + yAbsUp];
            data1 |= (uint)lifeData[x + yAbs];
            data2 |= (uint)lifeData[x + yAbsDown];

            uint result = 0;
            for (uint j = 0; j < 8; ++j) {
                uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
                aliveCells >>= 14;
                aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u)
                    + ((data2 >> 15) & 0x1u);

                result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

                data0 <<= 1;
                data1 <<= 1;
                data2 <<= 1;
            }
            resultLifeData[oldX + yAbs] = result;
        }
    }
}


bool runBitEncodeCUDAKernel(uint8_t* &d_gridOneEncoded, uint8_t* &d_gridTwoEncoded, size_t worldWidth, size_t worldHeight, 
    size_t iterationsCount, ushort threadsCount, uint bytesPerThread){

    if (worldWidth % 8 != 0) {
        return false;
    }
    size_t worldEncDataWidth = worldWidth / 8;
    if (worldEncDataWidth % bytesPerThread != 0) {
        return false;
    }    
    size_t encWorldSize = worldEncDataWidth * worldHeight;
    if ((encWorldSize / bytesPerThread) % threadsCount != 0) {
        return false;
    }

    size_t reqBlocksCount = (encWorldSize / bytesPerThread) / threadsCount;
    size_t blocksCount = std::min(size_t(32768), reqBlocksCount);

    for (size_t i = 0; i < iterationsCount; ++i) {
        bitLifeKernelCounting<<<blocksCount, threadsCount>>>(d_gridOneEncoded, uint(worldEncDataWidth),
            uint(worldHeight), bytesPerThread, d_gridTwoEncoded);
        std::swap(d_gridOneEncoded, d_gridTwoEncoded);
    }
    cudaDeviceSynchronize();
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