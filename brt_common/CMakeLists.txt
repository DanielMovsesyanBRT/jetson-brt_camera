# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE EQUAL Debug)
  set(CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c++ Debug flags" )
  set(CMAKE_C_FLAGS_DEBUG       "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c Debug flags" )
endif()

project(brt_common C CXX)

set(SOURCES 
  private/meta_impl.cpp
  metadata.cpp
  utils.cpp
  value_data.cpp
)
  
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME} 
                            PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
                            PRIVATE private)
