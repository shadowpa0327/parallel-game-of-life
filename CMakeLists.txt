cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Parallel-Game-Of-Life LANGUAGES CXX)

find_package(OpenMP)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(glm)
find_package(GLEW REQUIRED)


include_directories(
${CUDA_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -lpthread -fopenmp")
#string(APPEND CUDA_NVCC_FLAGS " -gencode arch=compute_86,code=sm_86")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 ")


add_library(utils
    utils.cpp
)  

cuda_add_executable(benchmark
    benchmark.cpp
)

cuda_add_executable(main
    main.cpp
)

target_link_libraries(benchmark
    utils
)


target_link_libraries(main
    utils
)


# -------- CUDA --------
cuda_add_library(gameOfLifeCUDA
    CUDAFunction.cu
    gameOfLifeCUDA.cpp
)
target_link_libraries(benchmark
    gameOfLifeCUDA
)
target_link_libraries(main
    gameOfLifeCUDA
)
# ----------------------

# ------- CPU ----------
add_library(gameOfLifeCPU
    gameOfLifeCPU.cpp
)
target_link_libraries(benchmark
    gameOfLifeCPU
)
target_link_libraries(main
    gameOfLifeCPU
)
# ----------------------


target_link_libraries(benchmark
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${glm_LIBRARY}
    ${GLUE_LIBRARY}
    ${GLEW_LIBRARIES}
    #OpenMP::OpenMP_CXX
)

target_link_libraries(main
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${glm_LIBRARY}
    ${GLUE_LIBRARY}
    ${GLEW_LIBRARIES}
    #OpenMP::OpenMP_CXX
)