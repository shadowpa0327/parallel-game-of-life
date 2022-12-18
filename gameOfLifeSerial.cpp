#include "parameter.h"
#include <algorithm>
#include <iostream>
void swap_mem(bool ***array1, bool ***array2){
    bool **tmp;
    tmp = *array1;
    *array1 = *array2;
    *array2 = tmp;
}

void gameOfLifeSerial(bool** &gridOne, bool** &gridTwo){
    // swap pointer
    // bool** temp = gridOne;
    // gridOne = gridTwo;
    // gridTwo = temp;
    std::swap(gridOne, gridTwo);
    for(int a = 1; a < gridHeight; a++)
    {
        for(int b = 1; b < gridWidth; b++)
        {
            int alive = gridTwo[a-1][b-1]   + gridTwo[a][b-1] + gridTwo[a+1][b-1]
                        + gridTwo[a-1][b]                     + gridTwo[a+1][b]
                        + gridTwo[a-1][b+1] + gridTwo[a][b+1] + gridTwo[a+1][b+1];;
            gridOne[a][b] = ((alive == 3) || (alive==2 && gridTwo[a][b]))?1:0; 
        }
    }
}
