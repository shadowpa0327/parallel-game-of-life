#include "parameter.h"

void gameOfLifeCUDA(bool** &gridOne, bool** &gridTwo);
void gameOfLifeOpenMP(bool** &gridOne, bool** &gridTwo);
void gameOfLifePthread(bool** &gridOne, bool** &gridTwo);
void gameOfLifeSerial(bool** &gridOne, bool** &gridTwo);