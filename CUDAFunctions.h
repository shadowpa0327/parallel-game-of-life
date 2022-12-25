#ifndef CUDA_FUNC_
#define CUDA_FUNC_

#include <cuda.h>


bool runSimpleKernel(uint8_t*& gridOne, uint8_t*& gridTwo, size_t gridWidth, size_t gridHeight, size_t iterationsCount, ushort threadsCount);
bool runBitEncodedKernel(uint8_t*& gridOne, uint8_t*& gridTwo, size_t gridWidth, size_t gridHeight);
bool runBitLifeEncodeKernel(const uint8_t* d_lifeData, int worldWidth, int worldHeight, uint8_t* d_encodedLife);
bool runBitLifeDecodeKernel(const uint8_t* d_encodedLife, uint worldWidth, uint worldHeight, uint8_t* d_lifeData);
bool runBitEncodeCUDAKernel(uint8_t* &d_gridOneEncoded, uint8_t* &d_gridTwoEncoded, size_t worldWidth, size_t worldHeight, 
    size_t iterationsCount, ushort threadsCount, uint bytesPerThread);


void runDisplayLifeKernel(const uint8_t* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
        int destWidth, int destHeight, int displacementX, int displacementY, int zoom, bool simulateColors,
        bool cyclic, bool bitLife)



#endif