// Author: Mario Talevski
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include "parameter.h"
#include "utils.h"
#include "gameOfLife.h"



#if defined(OS_WIN)
  #include <windows.h> // Use for windows
#endif

using namespace std;


int main(){
    cout << COLOR_GREEN;
    clearScreen();
    bool* gridOne = malloc_host<bool>(gridHeight*gridWidth, false);
    bool* gridTwo = malloc_host<bool>(gridHeight*gridWidth, false);
    bool* gridAns = malloc_host<bool>(gridHeight*gridWidth, false);

    char mode;

    cout << "Select Initialization mode, read from file or random sample (r/s): ";
    cin >> mode;

    initGrid(mode, gridOne);
    printGrid(gridOne);
    
    string start;
    cout << "Grid setup is done. Start the game ? (y/n): ";
    cin >> start;
    if( start == "y" || start == "Y" ) 
    {
      double serialTime = gameOfLifeSerial(gridOne, gridTwo, mode);
      printf("[Game Of Life Serial]:\t\t[%.3f] ms\n", serialTime);
      
      // copy answer to gridAns
      for (int i=0; i<(gridHeight)*(gridWidth); i++)
      {
          gridAns[i] = gridOne[i];
      }

      if (SHOW) return 0;
      #if defined(BUILD_PTHREAD)
        double pthreadTime = gameOfLifePthread(gridOne, gridTwo, mode);
        if (correct(gridOne, gridAns)) {
          printf("[Game Of Life Pthread]:\t\t[%.3f] ms", pthreadTime);
          printf("\t\t%.3f times faster than serial version\n", serialTime/pthreadTime);
        } else {
          printf("[Game Of Life Pthread]:\t\tWrong Answer\n");
        }

      #endif
      #if defined(BUILD_OPENMP)
        double OpenMPTime = gameOfLifeOpenMP(gridOne, gridTwo, mode);
        if (correct(gridOne, gridAns)) {
          printf("[Game Of Life OpenMP]:\t\t[%.3f] ms", OpenMPTime);
          printf("\t\t%.3f times faster than serial version\n", serialTime/OpenMPTime);
        } else {
          printf("[Game Of Life OpenMP]:\t\tWrong Answer\n");
        }

      #endif
      #if defined(BUILD_CUDA)
        double CUDATime = gameOfLifeCUDA(gridOne, gridTwo, mode);
        if (correct(gridOne, gridAns)) {
          printf("[Game Of Life CUDA]:\t\t\t\t[%.3f] ms", CUDATime); 
          printf("\t\t%.3f times faster than serial version\n", serialTime/CUDATime);
        } else {
          printf("[Game Of Life CUDA]:\t\tWrong Answer\n");
        }

      #endif
      #if defined(BUILD_BIT_CUDA)
        double BitCUDATime = gameOfLifeCUDABitEnocode(gridOne, gridTwo, mode);
        if (correct(gridOne, gridAns)) {
          printf("[Game Of Life CUDA(Bit version)]:\t\t[%.3f] ms", BitCUDATime); 
          printf("\t\t%.3f times faster than serial version\n", serialTime/BitCUDATime);
        } else {
          printf("[Game Of CUDA(Bit version)]:\t\tWrong Answer\n");
        }

      #endif
    } 
    free(gridOne);
    free(gridTwo);
    cout << COLOR_RESET;
    //clearScreen();
    return 0;
}
