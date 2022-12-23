#ifndef _GAME_OF_LIFE
#define _GAME_OF_LIFE

#include "parameter.h"
double gameOfLifeCUDA(bool* &gridOne, bool* &gridTwo, char mode);
double gameOfLifeCUDABitEnocode(bool* &gridOne, bool* &gridTwo, char mode);
double gameOfLifeOpenMP(bool* &gridOne, bool* &gridTwo, char mode);
double gameOfLifePthread(bool* &gridOne, bool* &gridTwo, char mode);
double gameOfLifeSerial(bool* &gridOne, bool* &gridTwo, char mode);

#endif
