project(Parallel-Game-Of-Life)
cmake_minimum_required(VERSION 2.8.9)



add_executable(main
    main.cpp
)

if(BUILD_SERIAL)
    add_library(gameOfLifeSerial
        gameOfLifeSerial.cpp
    )
    target_link_libraries(main
        gameOfLifeSerial
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_SERIAL")
endif()

if(BUILD_PTHREAD)
    add_library(gameOfLifePthread
        gameOfLifePthread.cpp
    )
    target_link_libraries(main
        gameOfLifePthread
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_PTHREAD")
endif()

if(BUILD_OPENMP)
    add_library(gameOfLifeOpenMP
        gameOfLifeOpenMP.cpp
    )
    target_link_libraries(main
        gameOfLifeOpenMP
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_OPENMP")
endif()

if(BUILD_CUDA)
    add_library(gameOfLifeCUDA
        gameOfLifeCUDA.cpp
    )
    target_link_libraries(main
        gameOfLifeCUDA
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_CUDA")
endif()


