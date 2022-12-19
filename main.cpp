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

    srand( time(NULL) );

    // system( "color A" );//LGT green
    cout << COLOR_GREEN;
    clearScreen();
    bool* gridOne = malloc_host<bool>((gridHeight+1)*(gridWidth+1), false);
    bool* gridTwo = malloc_host<bool>((gridHeight+1)*(gridWidth+1), false);

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
      #if defined(BUILD_SERIAL)
        double serialTime = gameOfLifeSerial(gridOne, gridTwo, mode);
        printf("[Game Of Life Serial]:\t\t[%.3f] ms\n", serialTime);
        //if (SHOW) return 0;
      #endif
      #if defined(BUILD_PTHREAD)
        double pthreadTime = gameOfLifePthread(gridOne, gridTwo, mode);
        printf("[Game Of Life Pthread]:\t\t[%.3f] ms", pthreadTime);
        printf("\t\t%.3f times faster than serial version\n", serialTime/pthreadTime);
      #endif
      #if defined(BUILD_OPENMP)
        double OpenMPTime = gameOfLifeOpenMP(gridOne, gridTwo, mode);
        printf("[Game Of Life OpenMP]:\t\t[%.3f] ms", OpenMPTime);
        printf("\t\t%.3f times faster than serial version\n", serialTime/OpenMPTime);
        if (SHOW) return 0;
      #endif
      #if defined(BUILD_CUDA)
        double CUDATime = gameOfLifeCUDA(gridOne, gridTwo, mode);
        printf("[Game Of Life CUDA]:\t\t[%.3f] ms", CUDATime); 
        printf("\t\t%.3f times faster than serial version\n", serialTime/CUDATime);
      #endif
    } 
    else 
    {
      cout << COLOR_RESET;
      clearScreen();
      return 0;
    }
    printf("pointer address in gridOne:%p\n", gridOne);
    printf("pointer address in gridTwo:%p\n", gridTwo);
    free(gridOne);
    free(gridTwo);
}
