cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Parallel-Game-Of-Life LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#find_package(GLEW REQUIRED)


cuda_add_executable(screen
    screen.cpp
)

add_library(utils
    ../utils.cpp
)  

target_link_libraries(screen
    utils
)


set_target_properties(screen
                    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(screen
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
)
 

