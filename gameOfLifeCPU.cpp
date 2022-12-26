#include "gameOfLifeCPU.h"
#include <omp.h>
#define NUM_THREADS 2

struct Arg
{
    int thread_id;
    size_t start;
    size_t end;
    uint8_t *gridOne;
    uint8_t *gridTwo;
    size_t gridHeight;
    size_t gridWidth;
};

void gameOfLifeCPU::updateSerial(uint8_t* &gridOne, uint8_t* &gridTwo, size_t gridHeight, size_t gridWidth){
    std::swap(gridOne, gridTwo);
    for(size_t a = 0; a < gridHeight; a++)
    {  
        size_t a_prev = ((a + gridHeight - 1) % gridHeight) * gridWidth;
        size_t a_cur = a*gridWidth;
        size_t a_next = ((a + 1) % gridHeight) * gridWidth;

        for(size_t b = 0; b < gridWidth; b++)
        {  
            size_t b_prev = (b + gridWidth-1) % gridWidth;
            size_t b_cur =  b;
            size_t b_next = (b + 1) % gridWidth;

            int alive =   gridTwo[a_prev + b_prev]   + gridTwo[a_cur + b_prev] + gridTwo[a_next + b_prev]
                        + gridTwo[a_prev + b_cur]                              + gridTwo[a_next + b_cur]
                        + gridTwo[a_prev + b_next]   + gridTwo[a_cur + b_next] + gridTwo[a_next + b_next];
            gridOne[a_cur + b_cur] = ((alive == 3) || (alive==2 && gridTwo[a_cur + b_cur]))?1:0; 
        }
    }
}

void* pthreadWorker(void *arg){

    Arg *data = (Arg *)arg;

    size_t start = data->start;
    size_t end = data->end;
    uint8_t* gridOne = data->gridOne;
    uint8_t* gridTwo = data->gridTwo;
    size_t gridHeight = data->gridHeight;
    size_t gridWidth = data->gridWidth;

    for(size_t a = start; a < end; a++)
    {  
        size_t a_prev = ((a + gridHeight - 1) % gridHeight) * gridWidth;
        size_t a_cur = a*gridWidth;
        size_t a_next = ((a + 1) % gridHeight) * gridWidth;

        for(size_t b = 0; b < gridWidth; b++)
        {  
            size_t b_prev = (b + gridWidth-1) % gridWidth;
            size_t b_cur =  b;
            size_t b_next = (b + 1) % gridWidth;

            int alive =   gridTwo[a_prev + b_prev]   + gridTwo[a_cur + b_prev] + gridTwo[a_next + b_prev]
                        + gridTwo[a_prev + b_cur]                              + gridTwo[a_next + b_cur]
                        + gridTwo[a_prev + b_next]   + gridTwo[a_cur + b_next] + gridTwo[a_next + b_next];
            gridOne[a_cur + b_cur] = ((alive == 3) || (alive==2 && gridTwo[a_cur + b_cur]))?1:0; 
        }
    }
    pthread_exit((void *)0); 
}

void updatePthread(uint8_t* &gridOne, uint8_t* &gridTwo, size_t gridHeight, size_t gridWidth){
    std::swap(gridOne, gridTwo);

    //Create threads;
    int part = gridHeight / NUM_THREADS;
    assert((gridHeight % NUM_THREADS) == 0 );
    Arg arg[NUM_THREADS];
    pthread_t Thd[NUM_THREADS]; 
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for(int i = 0; i < NUM_THREADS; i++)
    {
        arg[i].thread_id = i;
        arg[i].start = (part * i);
        arg[i].end = part * (i + 1);
        arg[i].gridOne = gridOne;
        arg[i].gridTwo = gridTwo;
        arg[i].gridHeight = gridHeight;
        arg[i].gridWidth = gridWidth;
        pthread_create(&Thd[i], &attr, pthreadWorker, (void *)&arg[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        // 等待每一個 thread 執行完畢
        void *status;
        pthread_join(Thd[i], &status);

        //int start = arg[i].start;
        //int end = arg[i].end;
        // for(int j = start; j <= end; j++)
        // {
        //     gridOne[j] = arg[i].gridOne[j];
        // }
    }
}

void gameOfLifeCPU::updateOpenMP(uint8_t* &gridOne, uint8_t* &gridTwo, size_t gridHeight, size_t gridWidth){
    std::swap(gridOne, gridTwo);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for(size_t a = 0; a < gridHeight; a++)
        {  
            size_t a_prev = ((a + gridHeight - 1) % gridHeight) * gridWidth;
            size_t a_cur = a*gridWidth;
            size_t a_next = ((a + 1) % gridHeight) * gridWidth;
            
            #pragma omp for
            for(size_t b = 0; b < gridWidth; b++)
            {  
                size_t b_prev = (b + gridWidth-1) % gridWidth;
                size_t b_cur =  b;
                size_t b_next = (b + 1) % gridWidth;

                int alive =   gridTwo[a_prev + b_prev]   + gridTwo[a_cur + b_prev] + gridTwo[a_next + b_prev]
                            + gridTwo[a_prev + b_cur]                              + gridTwo[a_next + b_cur]
                            + gridTwo[a_prev + b_next]   + gridTwo[a_cur + b_next] + gridTwo[a_next + b_next];
                gridOne[a_cur + b_cur] = ((alive == 3) || (alive==2 && gridTwo[a_cur + b_cur]))?1:0; 
            }
        }
    }   
}


void gameOfLifeCPU::iterate(size_t iterations, size_t gridHeight, size_t gridWidth, int updateMode){
    if (updateMode == 1) {
        for(size_t i = 0; i < iterations; i++){
            updateSerial(gridOne, gridTwo, gridHeight, gridWidth);
        }
    } else if (updateMode == 2) {
        for(size_t i = 0; i < iterations; i++){
            updatePthread(gridOne, gridTwo, gridHeight, gridWidth);
        } 
    } else if (updateMode == 3) {
        for(size_t i = 0; i < iterations; i++){
            updateOpenMP(gridOne, gridTwo, gridHeight, gridWidth);
        } 
    }

}