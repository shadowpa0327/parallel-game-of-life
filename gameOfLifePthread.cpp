#include "parameter.h"
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>
#include<pthread.h>
#define NUM_THREADS 8
struct Arg
{
    int thread_id;
    int start;
    int end;
    bool *gridOne;
    bool *gridTwo;
};


void* updatePthread(void *arg){
    // Todo: Implementation Pthread Version Here
    Arg *data = (Arg *)arg;
    
    int start = data->start;
    int end = data->end;
    for(int a = start; a <= end; a++)
    {
        for(int b = 1; b <= gridWidth; b++)
        {
            int alive =   data->gridTwo[(a-1)*arrayWidth + b-1]   + data->gridTwo[a*arrayWidth + b-1] + data->gridTwo[(a+1)*arrayWidth + b-1]
                        + data->gridTwo[(a-1)*arrayWidth + b]                                  + data->gridTwo[(a+1)*arrayWidth + b]
                        + data->gridTwo[(a-1)*arrayWidth + b+1]   + data->gridTwo[a*arrayWidth + b+1] + data->gridTwo[(a+1)*arrayWidth + b+1];;
            data->gridOne[a*arrayWidth + b] = ((alive == 3) || (alive==2 && data->gridTwo[a*arrayWidth + b]))?1:0; 
        }
    }
    pthread_exit((void *)0); 
    
}

double gameOfLifePthread(bool* &gridOne, bool* &gridTwo, char mode){
    
    initGrid(mode, gridOne);
    cout << "Running gameOfLife in Pthread mode\n";
    int iter = 0;
    double elapseTime = 0.0;

    //Create threads;
    int part = gridHeight / NUM_THREADS;
    Arg arg[NUM_THREADS];
    pthread_t Thd[NUM_THREADS]; 
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    while(iter++ < maxIteration)
    {   
        
        std::swap(gridOne, gridTwo);
        for(int i = 0; i < NUM_THREADS; i++)
        {
            arg[i].thread_id = i;
            arg[i].start = (part * i) + 1 ;
            arg[i].end = part * (i + 1);
            arg[i].gridOne = gridOne;
            arg[i].gridTwo = gridTwo;
            double startTime = CycleTimer::currentSeconds();
            pthread_create(&Thd[i], &attr, updatePthread, (void *)&arg[i]);
            double endTime = CycleTimer::currentSeconds();
            elapseTime += (endTime - startTime) * 1000;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            // 等待每一個 thread 執行完畢
            void *status;
            double startTime = CycleTimer::currentSeconds();
            pthread_join(Thd[i], &status);

            int start = arg[i].start;
            int end = arg[i].end;
            for(int j = start; j <= end; j++)
            {
                gridOne[j] = arg[i].gridOne[j];
            }
            double endTime = CycleTimer::currentSeconds();
            elapseTime += (endTime - startTime) * 1000;
            
        }
    }

    return elapseTime;
}
