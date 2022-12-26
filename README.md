# Game of Life

C++ implementation of Conway's Game of Life.
Also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970.

The "game" is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input. One interacts with the Game of Life by creating an initial configuration and observing how it evolves, or, for advanced "players", by creating patterns with particular properties.


## Where to modified the thread of CPU parallel
Currently, we set the thread number of CPU as a fixed number. User can modify it at `gameOfLifeCPU.cpp`.

## How to build
```
mkdir build
cd build

cmake .. 

make
```
## How to Run Gui
```
cd build
./main
```

## How to Run Test
```
cd build
./benchmark
```