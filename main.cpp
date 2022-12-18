// Author: Mario Talevski
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <time.h> 
#include "CycleTimer.h"
#include "parameter.h"
#include "gameOfLifeSerial.cpp"
#include "gameOfLifePthread.cpp"
#include "gameOfLifeOpenMP.cpp"
#include "gameOfLifeCUDA.cpp"

//Move OS defines up here to be used in different places
#if defined(_WIN32) || defined(WIN32) || defined(__MINGW32__) || defined(__BORLANDC__)
 #define OS_WIN
 //WINDOWS COLORS 
  #define COLOR_RED SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED)
  #define COLOR_WARNING SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN) 

  #define COLOR_BLUE SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_BLUE)

  #define COLOR_RESET SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15)

#elif defined(linux) || defined(__CYGWIN__)
  #define OS_LINUX

  #define COLOR_RED "\033[31m"
  #define COLOR_GREEN "\033[32m" 
  #define COLOR_BLUE "\033[34m"
  #define COLOR_RESET "\033[0m"

#elif (defined(__APPLE__) || defined(__OSX__) || defined(__MACOS__)) && defined(__MACH__)//To ensure that we are running on a mondern version of macOS
  #define OS_MAC

  #define COLOR_RED "\033[31m"
  #define COLOR_GREEN "\033[32m" 
  #define COLOR_BLUE "\033[34m"
  #define COLOR_RESET "\033[0m"

#endif

#if defined(OS_WIN)
  #include <windows.h> // Use for windows
#endif

using namespace std;

void initGrid(char mode, bool** gridOne);
void printGrid(bool** gridOne);
void clearScreen(void);

int main(){

    srand( time(NULL) );

    // system( "color A" );//LGT green
    cout << COLOR_GREEN;
    clearScreen();
    bool** gridOne = (bool**) malloc((gridHeight+1)*sizeof(bool*));
    bool** gridTwo = (bool**) malloc((gridHeight+1)*sizeof(bool*));
    for (int i=0; i<gridHeight+1; i++) 
    {
      gridOne[i] = (bool*) malloc((gridWidth+1)*sizeof(bool));
      gridTwo[i] = (bool*) malloc((gridWidth+1)*sizeof(bool));
    }

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
      int iter = 0;
      double serialTime = 0.0;
      while (iter++ < maxIteration) 
      {
        double startTime = CycleTimer::currentSeconds();
        gameOfLifeSerial(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        serialTime += (endTime - startTime) * 1000;
        
        if (SHOW) 
        {        
          printf("Iteration %d\n", iter);  
          printGrid(gridOne); 
          usleep(200000);
          clearScreen();
        }
	    }
      printf("[Game Of Life Serial]:\t\t[%.3f] ms\n", serialTime);

      if (SHOW) return 0;

      initGrid(mode, gridOne);
      iter = 0;
      double pthreadTime = 0.0;
      while (iter++ < maxIteration) 
      {
        double startTime = CycleTimer::currentSeconds();
        gameOfLifePthread(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        pthreadTime += (endTime - startTime) * 1000;
      }
      printf("[Game Of Life Pthread]:\t\t[%.3f] ms", pthreadTime);
      printf("\t\t%.3f times faster than serial version\n", serialTime/pthreadTime);

      initGrid(mode, gridOne);
      iter = 0;
      double OpenMPTime = 0.0;
      while (iter++ < maxIteration) 
      {
        double startTime = CycleTimer::currentSeconds();
        gameOfLifeOpenMP(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        OpenMPTime += (endTime - startTime) * 1000;
      }
      printf("[Game Of Life OpenMP]:\t\t[%.3f] ms", OpenMPTime);
      printf("\t\t%.3f times faster than serial version\n", serialTime/OpenMPTime);

      initGrid(mode, gridOne);
      iter = 0;
      double CUDATime = 0.0;
      while (iter++ < maxIteration) 
      {
        double startTime = CycleTimer::currentSeconds();
        gameOfLifeCUDA(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        CUDATime += (endTime - startTime) * 1000;
      }
      printf("[Game Of Life CUDA]:\t\t[%.3f] ms", CUDATime); 
      printf("\t\t%.3f times faster than serial version\n", serialTime/CUDATime);
    } 
    else 
    {
      cout << COLOR_RESET;
      clearScreen();
      return 0;
    }   
}

void initGrid(char mode, bool** gridOne) {
    for(int a = 0; a < gridHeight; a++)
    {
        for(int b = 0; b < gridWidth; b++)
        {
            gridOne[a][b] = false;
        }
    }

    int x, y;
    if (mode == 'r') 
    {
      string filename = "testboard.txt";  
      ifstream readfile(filename);
      if ( readfile.is_open() ) 
      {
        
        string fileline, xx, yy;

        while (getline(readfile,fileline)) 
        {
          stringstream ss(fileline);
          
          getline(ss, xx, ' ');
          getline(ss, yy, ' ');

          x = stoi(xx);
          y = stoi(yy);

          gridOne[x][y] = true;
        }
      }
    } 
    else if (mode == 's')
    {
      for(int a =0; a < gridHeight; a++)
      {
          for(int b = 0; b < gridWidth; b++)
          { 
            double sample = (double) rand() / RAND_MAX;
            if (sample < PROB) gridOne[a][b] = true;
          }
      }
    }
}

void clearScreen(void) {
  // Tested and working on Ubuntu and Cygwin
  #if defined(OS_WIN)
    HANDLE hStdOut = GetStdHandle( STD_OUTPUT_HANDLE );
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    DWORD count;
    DWORD cellCount;
    COORD homeCoords = { 0, 0 };

    if (hStdOut == INVALID_HANDLE_VALUE) return;

    /* Get the number of cells in the current buffer */
    GetConsoleScreenBufferInfo( hStdOut, &csbi );
    cellCount = csbi.dwSize.X *csbi.dwSize.Y;

    /* Fill the entire buffer with spaces */f
    FillConsoleOutputCharacter(hStdOut, (TCHAR) ' ', cellCount, homeCoords, &count);

    /* Fill the entire buffer with the current colors and attributes */
    FillConsoleOutputAttribute(hStdOut, csbi.wAttributes, cellCount, homeCoords, &count);

    SetConsoleCursorPosition( hStdOut, homeCoords );

  #elif defined(OS_LINUX) || defined(OS_MAC)
    cout << "\033[2J;" << "\033[1;1H"; // Clears screen and moves cursor to home pos on POSIX systems
  #endif

}

void printGrid(bool** gridOne){
    for(int a = 1; a < gridHeight; a++)
    {
        for(int b = 1; b < gridWidth; b++)
        {
            if (gridOne[a][b] == true)
            {
                cout << "█";
            }
            else
            {
                cout << ".";
            }
            if (b == gridWidth-1)
            {
                cout << endl;
            }
        }
    }
}
