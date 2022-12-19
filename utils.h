#ifndef ADD_UTIL
#define ADD_UTIL
#include <iostream>
#include <sstream>
#include <fstream>


#if defined(BUILD_CUDA)
#include <cuda.h>
#endif

using namespace std;

// Get position of array based on matrix indexes

#if defined(BUILD_CUDA)
static void cuda_check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}
#endif



///////////////////////////////////////////////////////////////////////////////
// Plotting
///////////////////////////////////////////////////////////////////////////////
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

void printGrid(bool* gridOne){
    for(int a = 1; a < gridHeight; a++)
    {
        for(int b = 1; b < gridWidth; b++)
        {
            if (gridOne[a*gridWidth+b] == true)
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


///////////////////////////////////////////////////////////////////////////////
// initialize the grid
///////////////////////////////////////////////////////////////////////////////
void initGrid(char mode, bool* gridOne) {

    // refill the grid
    std::fill(gridOne, gridOne+(gridHeight*gridWidth), false);

    int x, y;
    if (mode == 'r') 
    {
      // Todo: Try to do it in an more elastic way rather than hard coding it.
      string filename = "../testboard.txt";  
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

          gridOne[x*gridWidth + y] = true;
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
            if (sample < PROB) gridOne[a*gridWidth + b] = true;
          }
      }
    }
}



///////////////////////////////////////////////////////////////////////////////
// allocating memory
///////////////////////////////////////////////////////////////////////////////

template <typename T>
T* malloc_host(size_t N, T value=T()) {
    T* ptr = (T*)(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);

    return ptr;
}

#if defined(BUILD_CUDA)
// allocate space on GPU for n instances of type T
template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    cuda_check_status(status);
    return (T*)p;
}


// allocate managed memory
template <typename T>
T* malloc_managed(size_t n, T value=T()) {
    T* p;
    auto status = cudaMallocManaged(&p, n*sizeof(T));
    cuda_check_status(status);
    std::fill(p, p+n, value);
    return p;
}

template <typename T>
T* malloc_pinned(size_t N, T value=T()) {
    T* ptr = nullptr;
    cudaHostAlloc((void**)&ptr, N*sizeof(T), 0);

    std::fill(ptr, ptr+N, value);

    return ptr;
} 

///////////////////////////////////////////////////////////////////////////////
// copying memory
///////////////////////////////////////////////////////////////////////////////


// copy n*T from host to device
template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    cuda_check_status( cudaMemcpy(to,from,n*sizeof(T),cudaMemcpyHostToDevice) );
    // cudaFree(from);
}

// copy n*T from device to host
template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    cuda_check_status( cudaMemcpy(to,from,n*sizeof(T),cudaMemcpyDeviceToHost) );
    // cudaFree(from);
}
#endif

#endif
