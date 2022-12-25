#ifndef _GAME_OF_LIFE_CUDA
#define _GAME_OF_LIFE_CUDA

#include <cuda.h>

class gameOfLifeCUDA{
    private:
        uint8_t* d_gridOne;
        uint8_t* d_gridTwo;
        uint8_t* d_gridOneEncoded;
        uint8_t* d_gridTwoEncoded;

        size_t worldWidth;
        size_t worldHeight;

    public:
        gameOfLifeCUDA(){
            d_gridOne = nullptr;
            d_gridTwo = nullptr;
            d_gridOneEncoded = nullptr;
            d_gridTwoEncoded = nullptr;
            worldWidth = 0;
            worldHeight = 0;
        }

        uint8_t* getGridData() const{
            return d_gridOne;
        }
        uint8_t* getEncodedGridData() const{
            return d_gridOneEncoded;
        }

        void allocBuffers(bool encoded){
            freeBuffers();
            if(encoded){
                size_t worldSize = (worldWidth / 8) * worldHeight;
                cudaMalloc(&d_gridOneEncoded, worldSize);
                cudaMalloc(&d_gridTwoEncoded, worldSize);
            }
            else{
                size_t worldSize = worldWidth * worldHeight;
                cudaMalloc(&d_gridOne, worldSize);
                cudaMalloc(&d_gridTwo, worldSize);
            }
        }
        void freeBuffers(){
            cudaFree(d_gridOne);
            cudaFree(d_gridTwo);
            cudaFree(d_gridOneEncoded);
            cudaFree(d_gridTwoEncoded);
            d_gridOne = nullptr;
            d_gridTwo = nullptr;
            d_gridOneEncoded = nullptr;
            d_gridTwoEncoded = nullptr;
        }

        bool areBuffersAllocated(bool encoded){
            if(encoded){
                return d_gridOne != nullptr && d_gridTwo != nullptr;
            }
            else{
                return d_gridOneEncoded != nullptr && d_gridTwoEncoded != nullptr;
            }
        }

        void resizeWorld(size_t width, size_t height){
            freeBuffers();
            worldWidth = width;
            worldHeight = height;
        }

        void initWorld(uint8_t* data, bool encoded);

        void iterate(size_t iterations, bool bitLife, int threadsCount, int bitLifeBytesPerThread)
};





#endif
