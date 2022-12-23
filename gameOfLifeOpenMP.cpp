#include "parameter.h"
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <omp.h>
#include <unistd.h>
void updateOpenMP(bool* &gridOne, bool* &gridTwo){
    // Todo: Implementation OpenMP Version Here
    
    std::swap(gridOne, gridTwo);
    #pragma omp parallel num_threads(4)
    {
        //int num = omp_get_max_threads();
        //printf("thread num: %d\n",num);
        #pragma omp for 
        for(int a = 1; a <= gridHeight; a++)
        {
            //#pragma omp for private(b)
            for(int b = 1; b <= gridWidth; b++)
            {
                int alive =   gridTwo[(a-1)*arrayWidth + b-1]   + gridTwo[a*arrayWidth + b-1] + gridTwo[(a+1)*arrayWidth + b-1]
                            + gridTwo[(a-1)*arrayWidth + b]                                  + gridTwo[(a+1)*arrayWidth + b]
                            + gridTwo[(a-1)*arrayWidth + b+1]   + gridTwo[a*arrayWidth+ b+1] + gridTwo[(a+1)*arrayWidth + b+1];;
                gridOne[a*arrayWidth + b] = ((alive == 3) || (alive==2 && gridTwo[a*arrayWidth + b]))?1:0; 
            }
        }
    }
    
}

double gameOfLifeOpenMP(bool* &gridOne, bool* &gridTwo, char mode){
    
    initGrid(mode, gridOne);
    cout << "Running gameOfLife in OpenMP mode\n";
    int iter = 0;
    double elapseTime = 0.0;
    while (iter++ < maxIteration) 
    {
        double startTime = CycleTimer::currentSeconds();
        updateOpenMP(gridOne, gridTwo);
        double endTime = CycleTimer::currentSeconds();
        elapseTime += (endTime - startTime) * 1000;
        if (SHOW) 
        {        
            printf("Iteration (OpenMP)%d\n", iter);  
            printGrid(gridOne); 
            usleep(200000);
            clearScreen();
        }
    }
    return elapseTime;
}
