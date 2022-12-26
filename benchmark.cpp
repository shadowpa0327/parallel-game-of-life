#include <iostream>
#include <chrono>
#include <algorithm>
// CUDA and GameOfLife correlated header
#include "gameOfLifeCPU.h"
#include "gameOfLifeCUDA.h"
#include "CUDAFunctions.h"
#include "utils.h"
#include "parameter.h"
#include <iomanip>

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
uint8_t* answerGrid = nullptr;
uint8_t* temperGrid = nullptr;
char initializationMode; // mode for initialization


// Global Definition
enum cpuMode {Pthread, OpenMp};
enum cudaMode {Simple, BitEncode};
float referenceTime = 0.0;
char skipCPUTesting = 'y';

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
  answerGrid = new uint8_t[worldWidth * worldHeight];
  temperGrid = new uint8_t[worldWidth * worldHeight];
  initGrid(initializationMode, globalGrid);
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
    //cout << "Init the local buffer for CPU\n";
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
    //cout << "Initialize the buffer for GPU life\n";
    freeLocalBuffers();
    cudaLife.allocBuffers(bitLife);
    initWorld(true, bitLife);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  cudaLife.iterate(iterations, bitLife, threadsCount, bitLifeBytesPerTrhead);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
}


void cpuTest(cpuMode mode){
  cout<<endl;
  freeLocalBuffers();
  float time;
  if(mode == Pthread){
    cout << "[Running Benchmark]: Device: CPU,  mode: Pthread\n";
    time = runCpuLife(maxIteration, 2);
  }
  else{ // OpenMp
    cout << "[Running Benchmark]: Device: CPU,  mode: OpenMP\n";
    time = runCpuLife(maxIteration, 3);
  }
  
  cout << setw(14) <<"\t * execution time: "<< setw(6) <<fixed << setprecision(2) << time << " ms" << endl;
  memcpy(temperGrid, cpuLife.getGridData(), worldHeight * worldWidth);
  cout << setw(14) <<"\t * correctness: " << (correct(answerGrid, temperGrid)? "correct" : "error") << endl;
  cout << setw(14) << "\t * speedUp: " <<fixed << setprecision(2) << (referenceTime / time) <<"x" << endl;
}


void cudaTest(cudaMode mode){
  cout << endl;
  freeLocalBuffers();
  float time;
  if(mode == Simple){
    cout << "[Running Benchmark]: Device: GPU,  mode: Simple\n";
    time = runCUDALife(maxIteration, threadsCount, false, bitLifeBytesPerTrhead);
  }
  else{ // OpenMp
    cout << "[Running Benchmark]: Device: CPU,  mode: Bit Encode\n";
    time = runCUDALife(maxIteration, threadsCount, true, bitLifeBytesPerTrhead);
  }
  
  cout <<"\t * execution time: "<< setw(6) <<fixed << setprecision(2) << time << " ms" << endl;
  cudaLife.copyDataToCPU(temperGrid, mode == BitEncode);
  if(skipCPUTesting == 'n'){
    cout << setw(14) << "\t * correctness: " << (correct(answerGrid, temperGrid)? "correct" : "error") << endl;
    cout << setw(14) << "\t * speedUp: " <<fixed << setprecision(2) << (referenceTime / time) <<"x" << endl ;
  }
}


int main(){
    cout << "Select Initialization mode, read from file or random sample (r/s): ";
    cin >> initializationMode;
    assert((initializationMode == 'r' || initializationMode == 's'));
    cout << "Skip CPU testing for faster evaluation (y/n): ";
    cin >> skipCPUTesting;
    assert((skipCPUTesting == 'y' || skipCPUTesting == 'n'));
  	resizeLifeWorld(newWorldWidth, newWorldHeight);
    
    
    initGlobalGrid();


    freeLocalBuffers();

    if(skipCPUTesting == 'n'){
      referenceTime = runCpuLife(maxIteration, 1);
      cout <<endl <<  "[Running Referemce]: Device: CPU,  mode: Serial\n";
      cout <<"\t * reference time: "<< setw(6) <<fixed << setprecision(2) << referenceTime << " ms" << endl;
      // copy data as the reference answer
      memcpy(answerGrid, cpuLife.getGridData(), worldHeight * worldWidth);
      cpuTest(cpuMode::Pthread);
      cpuTest(cpuMode::OpenMp);
    }
    cudaTest(cudaMode::Simple);
    cudaTest(cudaMode::BitEncode);
    cout << endl;
    return 0;
}
