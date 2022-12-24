cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Parallel-Game-Of-Life LANGUAGES CUDA CXX)

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

cuda_add_executable(screen
    screen.cpp
)

# Utils
#-----------------
add_library(utils
    ../utils.cpp
)  

target_link_libraries(screen
    utils
)
#------------------


# CUDA
#------------------
cuda_add_library(gameOfLifeCUDA
    display.cu
)
#------------------




set_target_properties(screen
                    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                    
target_link_libraries(screen
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${glm_LIBRARY}
    ${GLUE_LIBRARY}
    ${GLEW_LIBRARIES}
)
 

