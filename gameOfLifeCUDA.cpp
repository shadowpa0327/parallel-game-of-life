#include "CUDAFunctions.h"
#include "gameOfLifeCUDA.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
void gameOfLifeCUDA::iterate(size_t iterations, bool bitLife, int threadsCount, int bitLifeBytesPerThread){
    if(bitLife){
        return runBitEncodeCUDAKernel(d_gridOneEncoded, d_gridTwoEncoded, worldWidth,
                                      worldHeight, iterations, threadsCount, bitLifeBytesPerThread);
    }
    else{
        return runSimpleLifeKernel(d_gridOne, d_gridTwo, worldWidth, worldHeight, iterations, threadsCount);
    }
}


void gameOfLifeCUDA::initWorld(uint8_t* data, bool encoded){
    int worldSize = worldWidth * worldHeight;
    if(encoded){
        // generate the encoded data and write it to the device buffer reserve for encoded world.
        assert(areBuffersAllocated(encoded));
        uint8_t* d_data;
        cudaHostRegister(data, worldSize*sizeof(uint8_t), cudaHostRegisterMapped);
        cudaHostGetDevicePointer(&d_data, data, 0);
        runBitLifeEncodeKernel(d_data, worldWidth, worldHeight, d_gridOneEncoded);
        cudaHostUnregister(data);
    }
    else{
        assert(areBuffersAllocated(encoded));
        cudaMemcpy(d_gridOne, data, worldSize, cudaMemcpyHostToDevice);
    }
}


void gameOfLifeCUDA::copyDataToCPU(uint8_t* destination, bool encoded){
    if(encoded){
        uint8_t* tmp_buf;
        cudaMalloc((void**)&tmp_buf, worldHeight * worldWidth);
        runBitLifeDecodeKernel(d_gridOneEncoded, worldWidth, worldHeight, tmp_buf);
        cudaMemcpy(destination, tmp_buf, worldWidth * worldHeight ,cudaMemcpyDeviceToHost);
    }
    else{
        cudaMemcpy(destination, d_gridOne, worldWidth * worldHeight ,cudaMemcpyDeviceToHost);
    }
}