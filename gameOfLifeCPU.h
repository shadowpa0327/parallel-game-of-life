#ifndef _GAME_OF_LIFE_CUDA
#define _GAME_OF_LIFE_CUDA

#include <cuda.h>
#include "utils.h"

class gameOfLifeCPU{
    private:
        uint8_t* gridOne;
        uint8_t* gridTwo;

        size_t worldWidth;
        size_t worldHeight;

    public:
        gameOfLifeCUDA(){
            gridOne = nullptr;
            gridTwo = nullptr;
            worldWidth = 0;
            worldHeight = 0;
        }

        uint8_t* getGridData() const{
            return gridOne;
        }
        uint8_t* getEncodedGridData() const{
            return gridOneEncoded;
        }

        void allocBuffers(){
            freeBuffers();
            size_t worldSize = worldWidth * worldHeight;
            gridOne = new uint8[worldSize];
            gridTwo = new uint8[worldSize];
        }

        void freeBuffers(){
            delete[] gridOne;
            delete[] gridTwo;
            gridOne = nullptr;
            gridTwo = nullptr;
        }

        void resizeWorld(size_t width, size_t height){
            freeBuffers();
            worldWidth = width;
            worldHeight = height;
        }

        bool areBuffersAllocated(){
            return (gridOne != nullptr && gridTwo != nullptr);
        }
        
        void initWorld(uint8_t* data){
            assert(areBuffersAllocated());
            int worldSize = worldWidth * worldHeight;
            memcpy(gridOne, data, worldSize);
        }



        void iterate(size_t iterations);
    private:
        void updateSerial(uint8_t* &gridOne, uint8_t* &gridTwo, size_t gridHeight, size_t gridWidth);
};





#endif
