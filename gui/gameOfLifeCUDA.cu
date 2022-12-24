#include <cuda_runtime.h>
#include <assert.h>
#include "../utils.h"
#include "../parameter.h"


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

void runGameOfLifeCUDA(bool* &d_gridOne, bool* &d_gridTwo){
    int size = gridWidth * gridHeight;
	std::swap(d_gridOne, d_gridTwo);
	size_t threadCount = min(128, size);
	size_t reqBlocksCount = ((gridWidth) * (gridHeight)) / threadCount;
	updateCUDAKernel<<<reqBlocksCount, threadCount>>>(d_gridOne, d_gridTwo);
	printf("run updateCUDAKernel : %s\n", cudaGetErrorString(cudaGetLastError())); 
}