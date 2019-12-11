# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

#if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#  #set(CMAKE_TOOLCHAIN_FILE /home/daniel/Development/ws/JupiterEmbedded/driveworks/cmake/Toolchain-V5L.cmake)
#  message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE must be specified")
#endif()

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE EQUAL Debug)
  set(CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c++ Debug flags" )
  set(CMAKE_C_FLAGS_DEBUG       "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c Debug flags" )
endif()

project(script C CXX)

set(SOURCES 
  expression.cpp
  function.cpp
  parser.cpp
  parser_string.cpp
)

add_library(${PROJECT_NAME} STATIC ${SOURCES})
# add_dependencies(${PROJECT_NAME} brt_common)
target_include_directories(${PROJECT_NAME} 
                          PUBLIC 
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
                             
target_link_libraries(${PROJECT_NAME} brt_common) 
