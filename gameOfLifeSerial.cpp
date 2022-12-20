#include "parameter.h"
#include <algorithm>
#include <iostream>
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>

void updateSerial(bool* &gridOne, bool* &gridTwo){
    std::swap(gridOne, gridTwo);
    for(int a = 1; a <= gridHeight; a++)
    {
        for(int b = 1; b <= gridWidth; b++)
        {
            int alive =   gridTwo[(a-1)*arrayWidth + b-1]   + gridTwo[a*arrayWidth + b-1] + gridTwo[(a+1)*arrayWidth + b-1]
                        + gridTwo[(a-1)*arrayWidth + b]                                  + gridTwo[(a+1)*arrayWidth + b]
                        + gridTwo[(a-1)*arrayWidth + b+1]   + gridTwo[a*arrayWidth + b+1] + gridTwo[(a+1)*arrayWidth + b+1];;
            gridOne[a*arrayWidth + b] = ((alive == 3) || (alive==2 && gridTwo[a*arrayWidth + b]))?1:0; 
        }
    }
}

double gameOfLifeSerial(bool* &gridOne, bool* &gridTwo, char mode){
    initGrid(mode, gridOne);
    cout <<"Running gameOfLife in serial mode \n";
    int iter = 0;
    double elapseTime = 0.0;
    while (iter++ < maxIteration) 
    {
        double startTime = CycleTimer::currentSeconds();
        updateSerial(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        elapseTime += (endTime - startTime) * 1000;
        
        if (SHOW) 
        {        
            printf("Iteration %d\n", iter);  
            printGrid(gridOne); 
            usleep(200000);
            clearScreen();
        }
    }
    printGrid(gridOne); 
    return elapseTime;
}

