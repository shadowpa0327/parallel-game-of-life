#include "parameter.h"
#include <algorithm>
#include <iostream>
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>

void updateSerial(bool* &gridOne, bool* &gridTwo){
    std::swap(gridOne, gridTwo);
    for(int a = 0; a < gridHeight; a++)
    {  
        int a_prev = ((a + gridHeight - 1) % gridHeight) * gridWidth;
        int a_cur = a*gridWidth;
        int a_next = ((a + 1) % gridHeight) * gridWidth;

        for(int b = 0; b < gridWidth; b++)
        {  
            int b_prev = (b + gridWidth-1) % gridWidth;
            int b_cur =  b;
            int b_next = (b + 1) % gridWidth;

            int alive =   gridTwo[a_prev + b_prev]   + gridTwo[a_cur + b_prev] + gridTwo[a_next + b_prev]
                        + gridTwo[a_prev + b_cur]                              + gridTwo[a_next + b_cur]
                        + gridTwo[a_prev + b_next]   + gridTwo[a_cur + b_next] + gridTwo[a_next + b_next];
            gridOne[a_cur + b_cur] = ((alive == 3) || (alive==2 && gridTwo[a_cur + b_cur]))?1:0; 
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
            clearScreen();
            printf("Iteration %d\n", iter);  
            printGrid(gridOne); 
            usleep(300000);
        }
    } 
    return elapseTime;
}

