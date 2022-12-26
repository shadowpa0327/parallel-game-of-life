#ifndef _GAME_OF_LIFE_CUDA
#define _GAME_OF_LIFE_CUDA

#include <cuda.h>
#include <cassert>
#include "utils.h"
#include <cstring>
class gameOfLifeCPU{
    private:
        uint8_t* gridOne;
        uint8_t* gridTwo;

        size_t worldWidth;
        size_t worldHeight;

    public:
        gameOfLifeCPU(){
            gridOne = nullptr;
            gridTwo = nullptr;
            worldWidth = 0;
            worldHeight = 0;
        }

        uint8_t* getGridData() const{
            return gridOne;
        }

        void allocBuffers(){
            freeBuffers();
            size_t worldSize = worldWidth * worldHeight;
            gridOne = new uint8_t[worldSize];
            gridTwo = new uint8_t[worldSize];
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
            std::memcpy(gridOne, data, worldSize);
        }



        void iterate(size_t iterations, size_t gridHeight, size_t gridWidth, int updateMode);
    private:
        void updateSerial(uint8_t* &gridOne, uint8_t* &gridTwo, size_t gridHeight, size_t gridWidth);
        void updateOpenMP(uint8_t* &gridOne, uint8_t* &gridTwo, size_t gridHeight, size_t gridWidth);
};





#endif
