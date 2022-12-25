#include "utils.h"
#include "parameter.h"

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


void printGrid(uint8_t* gridOne){
    for(int a = 0; a < gridHeight; a++)
    {
        for(int b = 0; b < gridWidth; b++)
        {
            if (gridOne[a*gridWidth+b] == 1)
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

void initGrid(char mode, uint8_t* gridOne) {
    srand( 811514 );
    // refill the grid
    std::fill(gridOne, gridOne+(gridWidth*gridHeight), false);
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
          //cout<<x<< " " << y <<endl;
          gridOne[x*gridWidth + y] = 1;
        }
      }
      else{
        std::cout << "error\n" <<std::endl;
      }
    } 
    else if (mode == 's')
    {
      for(int a = 0; a < gridHeight; a++)
      {
          for(int b = 0; b < gridWidth; b++)
          { 
            double sample = (double) rand() / RAND_MAX;
            if (sample < PROB) gridOne[a*gridWidth + b] = true;
          }
      }
    }
}

bool correct(uint8_t* gridOne, uint8_t* gridAns) {
  for (int i=0; i<(gridHeight)*(gridWidth); i++) {
    if (gridOne[i] != gridAns[i]) return false;  
  }
  return true;
}
