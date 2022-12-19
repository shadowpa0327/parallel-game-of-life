cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Parallel-Game-Of-Life LANGUAGES CXX)


add_library(utils
    utils.cpp
)  


add_executable(main
    main.cpp
)

target_link_libraries(main
    utils
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_OPENMP -fopenmp")
endif()

if(BUILD_CUDA)

    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    add_library(gameOfLifeCUDA
        gameOfLifeCUDA.cu
    )
    #message(STATUS "${CUDA_INCLUDE_DIRS}")
    include_directories(${CUDA_INCLUDE_DIRS})
    set_target_properties(gameOfLifeCUDA
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(main
        gameOfLifeCUDA
        ${CUDA_LIBRARIES}
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_CUDA")
endif()


