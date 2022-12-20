
#include "parameter.h"
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>
#include <cuda.h>
__global__ void updateCUDAKernel(bool* gridOne, bool* gridTwo){
//__global__ void updateCUDAKernel(bool* gridOne, bool* gridTwo)
    uint worldSize = gridWidth * gridHeight;
    //printf("%d\n", worldSize);
    for(uint cellID = blockIdx.x * blockDim.x + threadIdx.x;
        cellID < worldSize;
        cellID += blockDim.x * gridDim.x){
        //printf("%d\n", cellID);
        uint x = (cellID % (gridWidth)) + 1; // 1 base
        uint y = ((cellID - (cellID % (gridWidth)))/gridWidth)*arrayWidth + arrayWidth;  // 1 base
        uint xLeft = x - 1;
        uint xRight = x + 1;
        uint yUp = y - arrayWidth;
        uint yDown = y + arrayWidth;

        uint alive =      gridTwo[xLeft + yUp]   + gridTwo[x + yUp]   + gridTwo[xRight + yUp] +
                          gridTwo[xLeft + y]     +                    + gridTwo[xRight + y] +
                          gridTwo[xLeft + yDown] + gridTwo[x + yDown] + gridTwo[xRight + yDown];
        
        gridOne[x + y] = alive == 3 || (alive == 2 && gridTwo[x + y]) ? 1 : 0 ;   
    }
//     uint cellID = blockIdx.x * blockDim.x + threadIdx.x;
//     uint x = (cellID % (gridWidth)) + 1; // 1 base
//     uint y = ((cellID - (cellID % (gridWidth)))/gridWidth)*arrayWidth + arrayWidth; // 1 base
//     uint xLeft = x - 1;
//     uint xRight = x + 1;
//     uint yUp = y - arrayWidth;
//     uint yDown = y + arrayWidth;
//    // printf("%d\n", yDown);
//     uint alive = gridTwo[xLeft + yUp]   + gridTwo[x + yUp]   + gridTwo[xRight + yUp] +
//                  gridTwo[xLeft + y]     +                    + gridTwo[xRight + y] +
//                  gridTwo[xLeft + yDown] + gridTwo[x + yDown] + gridTwo[xRight + yDown];
//     //uint alive = 2;
//     gridOne[x + y] = alive == 3 || (alive == 2 && gridTwo[x + y]) ? 1 : 0 ;   
//     //gridOne[x+y]=1;
}

double gameOfLifeCUDA(bool* &gridOne, bool* &gridTwo, char mode){
    
    int i = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    initGrid(mode, gridOne);
    

    int size = arrayHeight * arrayWidth;
    bool *d_gridOne, *d_gridTwo;
    cudaError_t status;
    
    cudaMalloc(&d_gridOne, size*sizeof(bool));
    cudaMalloc(&d_gridTwo, size*sizeof(bool));
    cudaMemcpy(d_gridOne, gridOne, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridTwo, gridTwo, size, cudaMemcpyHostToDevice);
    //usleep(200000000);
    int iter = 0;  
    float elapseTime = 0.0;
    size_t threadCount = min(128, size);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    while (iter++ < maxIteration) 
    {
        std::swap(d_gridOne, d_gridTwo);
        size_t reqBlocksCount = ((gridWidth) * (gridHeight)) / threadCount;
        updateCUDAKernel<<<reqBlocksCount, threadCount>>>(d_gridOne, d_gridTwo);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapseTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(gridOne, d_gridOne, size, cudaMemcpyDeviceToHost);
    printGrid(gridOne);
    cudaFree(d_gridOne);
    cudaFree(d_gridTwo);
    return elapseTime;
}
