#include "parameter.h"

void gameOfLifeOpenMP(bool** &gridOne, bool** &gridTwo){
    // swap pointer
    bool** temp = gridOne;
    gridOne = gridTwo;
    gridTwo = temp;

    for(int a = 1; a < gridHeight; a++)
    {
        for(int b = 1; b < gridWidth; b++)
        {
            int alive = 0;
            for(int c = -1; c < 2; c++)
            {
                for(int d = -1; d < 2; d++)
                {
                    if(!(c == 0 && d == 0))
                    {
                        if(gridTwo[a+c][b+d])
                        {
                          ++alive;
                        }
                    }
                }
            }
            if (alive < 2)
            {
                gridOne[a][b] = false;
            }
            else if (alive == 3)
            {
                gridOne[a][b] = true;
            }
            else if (alive > 3)
            {
                gridOne[a][b] = false;
            }
        }
    }
}
