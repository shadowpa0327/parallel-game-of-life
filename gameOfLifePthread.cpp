#include "parameter.h"
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>

void updatePthread(bool* &gridOne, bool* &gridTwo){
    // Todo: Implementation Pthread Version Here
}

double gameOfLifePthread(bool* &gridOne, bool* &gridTwo, char mode){
    
    initGrid(mode, gridOne);
    int iter = 0;
    double elapseTime = 0.0;
    while (iter++ < maxIteration) 
    {
        double startTime = CycleTimer::currentSeconds();
        updatePthread(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        elapseTime += (endTime - startTime) * 1000;
    }
    return elapseTime;
}
