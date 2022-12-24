#ifndef _DISPLAY
#define _DISPLAY

void runDisplayLifeKernel(const uint8_t* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
        int destWidth, int destHeight, int displacementX, int displacementY, int zoom, bool simulateColors,
        bool cyclic, bool bitLife);

#endif