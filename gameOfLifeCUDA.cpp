#include "CUDAFunctions.h"
#include "gameOfLifeCUDA.h"

void gameOfLifeCUDA::iterate(size_t iterations, bool bitLife, int threadsCount, int bitLifeBytesPerThread){
    if(bitLife){
        return runBitEncodeCUDAKernel(d_gridOneEncoded, d_gridTwoEncoded, worldWidth,
                                      worldHeight, iterations, threadsCount, bitLifeBytesPerThread);
    }
    else{
        return runSimpleKernel(d_gridOne, d_gridTwo, worldWidth, worldHeight, iterations, threadsCount);
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
        runBitEncodedKernel(d_data, worldWidth, worldHeight, d_gridOneEncoded);
        cudaHostUnregister(data);
    }
    else{
        assert(areBuffersAllocated(encoded));
        cudaMemcpy(d_gridOne, data, worldSize, cudaMemcpyHostToDevice);
    }
}