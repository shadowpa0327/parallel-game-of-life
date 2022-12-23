
#include "parameter.h"
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>
#include <cuda.h>
#include <cassert>

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


/// Runs a kernel that encodes byte-per-cell data to bit-per-cell data.
void runBitLifeEncodeKernel(const uint8_t* d_lifeData, int worldWidth, int worldHeight, uint8_t* d_encodedLife) {

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



/// CUDA kernel that decodes data from bit-per-cell to byte-per-cell format.
/// Needs to be invoked for each byte in encoded data (cells / 8).
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




double gameOfLifeCUDABitEnocode(bool* &gridOne, bool* &gridTwo, char mode){
    
    int i = 0;

    
    initGrid(mode, gridOne);

    int size = gridHeight * gridWidth;
    
    uint8_t* _gridOne = (uint8_t*)gridOne;
    uint8_t* d_gridOne;
    uint8_t* d_gridOneEncoded;
    uint8_t* d_gridTwoEncoded;
    float elapseTime = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

//    cudaMalloc(&d_gridOne, size*sizeof(uint8_t));
//    cudaMemcpy(d_gridOne, _gridOne, size*sizeof(uint8_t), cudaMemcpyHostToDevice); 
    cudaHostRegister(_gridOne, size*sizeof(uint8_t), cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&d_gridOne, _gridOne, 0);

    cudaMalloc(&d_gridOneEncoded, size*sizeof(uint8_t)/8);
    cudaMalloc(&d_gridTwoEncoded, size*sizeof(uint8_t)/8);

    runBitLifeEncodeKernel(d_gridOne, gridWidth, gridHeight, d_gridOneEncoded);



    runBitEncodeCUDAKernel(d_gridOneEncoded, d_gridTwoEncoded, gridWidth, gridHeight, maxIteration, 128, 1);

    runBitLifeDecodeKernel(d_gridOneEncoded, gridHeight, gridWidth, d_gridOne);
    //cudaMemcpy(gridOne, d_gridOne, size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaHostUnregister(_gridOne);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapseTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapseTime;

}
