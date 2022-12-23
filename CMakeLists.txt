cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Parallel-Game-Of-Life LANGUAGES CXX)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
string(APPEND CUDA_NVCC_FLAGS " -gencode arch=compute_86,code=sm_86")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 ")


add_library(utils
    utils.cpp
)  


cuda_add_executable(main
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
    set(MY_CUDA_SRC
        gameOfLifeCUDA.cu
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_CUDA")
    if(BUILD_BIT_CUDA)
        set(MY_CUDA_SRC ${MY_CUDA_SRC} gameOfLifeCUDABit.cu)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_BIT_CUDA")
    endif()
   
    cuda_add_library(gameOfLifeCUDA
        ${MY_CUDA_SRC}
    )
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(gameOfLifeCUDA
        utils
    )
    set_target_properties(gameOfLifeCUDA
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(main
        gameOfLifeCUDA
        ${CUDA_LIBRARIES}
    )
    
endif()


