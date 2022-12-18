#include "parameter.h"

void gameOfLifePthread(bool gridOne[gridHeight+1][gridWidth+1]){
    bool gridTwo[gridHeight+1][gridWidth+1] = {};
    for(int a = 0; a < gridHeight; a++)
    {
        for(int b = 0; b < gridWidth; b++)
        {
          gridTwo[a][b] = gridOne[a][b];
        }
    }

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
