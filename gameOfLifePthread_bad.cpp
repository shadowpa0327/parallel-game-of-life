#include "parameter.h"
#include "utils.h"
#include <time.h> 
#include "CycleTimer.h"
#include <unistd.h>
#include<pthread.h>
#define NUM_THREADS 2
struct Arg
{
    int thread_id;
    int start;
    int end;
    bool *gridOne;
    bool *gridTwo;
};
pthread_mutex_t mutexsum;


void* updatePthread(void *arg){
    // Todo: Implementation Pthread Version Here
    pthread_mutex_lock(&mutexsum);
    Arg *data = (Arg *)arg;
    std::swap(data->gridOne, data->gridTwo);

    // cout << "*******************gridOne**********************\n";
    // for(int i = 0; i < arrayHeight*arrayWidth; i++)
    // {
    //     cout << data->gridOne[i] << " ";
    // }
    // cout << "\n";
    // cout << "*******************gridTwo**********************\n";
    // for(int i = 0; i < arrayHeight*arrayWidth; i++)
    // {
    //     cout << data->gridTwo[i] << " ";
    // }
    // cout << "\n";
    // cout << "************************************************\n";
    
    int start = data->start;
    int end = data->end;
    // for(int i = 0; i < arrayHeight*arrayWidth; i++)
    // {
    //     cout << data->gridTwo[i] << " ";
    // }
    //cout << "\n";
    //cout << "Hello from Thread "<<data->thread_id << endl;
    cout << "id: "<< data->thread_id << " " << data->start << " " << data->end << endl;
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
    // for(int i = 0; i < arrayHeight*arrayWidth; i++)
    // {
    //     cout << data->gridOne[i] << " ";
    // }
    // cout << "\n";
    pthread_mutex_unlock(&mutexsum);
    pthread_exit((void *)0); 
    
}

double gameOfLifePthread(bool* &gridOne, bool* &gridTwo, char mode){
    
    initGrid(mode, gridOne);
    cout << "Running gameOfLife in Pthread mode\n";
    int iter = 0;
    double elapseTime = 0.0;

    //Create threads;
    pthread_mutex_init(&mutexsum, NULL);
    pthread_t Thd[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    int part = gridHeight / NUM_THREADS;
    Arg arg[NUM_THREADS];
    while(iter++ < maxIteration)
    {
        //cout << iter << '\n';
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
        //pthread_attr_destroy(&attr);
        for (int i = 0; i < NUM_THREADS; i++)
        {
            // 等待每一個 thread 執行完畢
            void *status;
            double startTime = CycleTimer::currentSeconds();
            pthread_join(Thd[i], &status);

            // cout << "Threads:\n";
            for(int j = 0; j < arrayHeight*arrayWidth; j++)
            {
                //cout << arg[i].gridOne[j] << " ";
                gridOne[j] = arg[i].gridOne[j];
                gridTwo[j] = arg[i].gridOne[j];
            }
            double endTime = CycleTimer::currentSeconds();
            elapseTime += (endTime - startTime) * 1000;
            // cout << "\n";
            
        }
        // cout << "Iteration:"<<iter<<"\n";
        // for(int i = 0; i < arrayHeight*arrayWidth; i++)
        // {
        //     cout << gridOne[i] << " ";
        // }
        // cout << "\n";
        //pthread_exit(NULL);

    }

    
    pthread_mutex_destroy(&mutexsum);
    return elapseTime;
}
