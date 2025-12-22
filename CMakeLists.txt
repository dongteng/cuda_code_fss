cmake_minimum_required(VERSION 3.16.3)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc") #得添加这句

project(CUDA_Course_Project  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-O3 ${CMAKE_CUDA_FLAGS}")
include(cmake/cuda.cmake)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(course1)
add_subdirectory(course2)
add_subdirectory(course3)
add_subdirectory(course4)
add_subdirectory(course6)
add_subdirectory(course5_1)
add_subdirectory(course7)
add_subdirectory(course8)
add_subdirectory(course9)

