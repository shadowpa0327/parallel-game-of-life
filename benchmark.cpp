#include <iostream>
#include <chrono>

// CUDA and GameOfLife correlated header
#include "gameOfLifeCPU.h"
#include "gameOfLifeCUDA.h"
#include "CUDAFunctions.h"
#include "utils.h"
#include "parameter.h"

// GameOfLifeCUDA settings
bool bitLife = true;
ushort threadsCount = 128;
uint bitLifeBytesPerTrhead = 1u;

// game of life settings
size_t worldWidth = gridWidth;
size_t worldHeight = gridHeight;

size_t newWorldWidth = gridWidth;
size_t newWorldHeight = gridHeight;

// GameOfLife Object of CPU and GPU
gameOfLifeCPU cpuLife;
gameOfLifeCUDA cudaLife;

// Global buffer for shared data
uint8_t* globalGrid = nullptr;
char initialization_mode; // mode for initialization

void freeLocalBuffers(){
  cpuLife.freeBuffers();
  cudaLife.freeBuffers();
}

void resizeLifeWorld(size_t newWorldWidth, size_t newWorldHeight){
  freeLocalBuffers();

  worldWidth = newWorldWidth;
  worldHeight = newWorldHeight;

  cpuLife.resizeWorld(worldWidth, worldHeight);
  cudaLife.resizeWorld(worldWidth, worldHeight);
}

void initGlobalGrid(){  
  globalGrid = new uint8_t[worldWidth * worldHeight];
  initGrid(initialization_mode, globalGrid);
}

void initWorld(bool isCUDA, bool bitlife){
  if(isCUDA){
    cudaLife.initWorld(globalGrid, bitlife);
  }
  else{
    cpuLife.initWorld(globalGrid);
  }
}

float runCpuLife(size_t iterations, int updateMode){
  if(!cpuLife.areBuffersAllocated()){
    freeLocalBuffers();
    cpuLife.allocBuffers();
    initWorld(false, false);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  cpuLife.iterate(iterations, worldHeight, worldWidth, updateMode);
  auto t2 = std::chrono::high_resolution_clock::now();
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

int main(){
    cout << "Select Initialization mode, read from file or random sample (r/s): ";
    cin >> initialization_mode;

  	resizeLifeWorld(newWorldWidth, newWorldHeight);
  
    /*
    initGlobalGrid();
    float time0 = runCpuLife(1000, 3);
    cout << time0 << endl;

    initGlobalGrid();
    float time = runCpuLife(1000, 3);
    cout << time << endl;

    initGlobalGrid();
    float time2 = runCpuLife(1000, 3);
    cout << time2 << endl;

    initGlobalGrid();
    float time3 = runCpuLife(1000, 3);
    cout << time3 << endl;
    */

    initGlobalGrid();
    float time4 = runCUDALife(100000, threadsCount, bitLife, bitLifeBytesPerTrhead);
    cout << time4 << endl;

    initGlobalGrid();
    float time5 = runCUDALife(100000, threadsCount, false, bitLifeBytesPerTrhead);
    cout << time5 << endl;

    return 0;
}
