# Copyright (c) 2020, Blueriver Corporation. All rights reserved.
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE EQUAL Debug)
  set(CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c++ Debug flags" )
  set(CMAKE_C_FLAGS_DEBUG       "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c Debug flags" )
endif()

# set(CMAKE_EXE_LINKER_FLAGS "-Wl,-as-needed")

project(cuda_lib VERSION 1.0 LANGUAGES CXX CUDA)

set(SOURCES 
  cuda_lib.cu
  cuda_debayer.cu
  cuda_2d_data.cu
  cuda_color_convert.cu
  cuda_data.cu
)
    
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME} 
                            PUBLIC 
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            PRIVATE)

target_link_libraries(${PROJECT_NAME} brt_common) 
