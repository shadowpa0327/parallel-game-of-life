#include "gameOfLifeCPU.h"



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



void gameOfLifeCPU::iterate(size_t iterations, size_t gridHeight, size_t gridWidth){
    // Todo Add pthread and OpenMP version here.
    for(size_t i = 0; i < iterations; i++){
        updateSerial(gridOne, gridTwo, gridHeight, gridWidth);
    }
}