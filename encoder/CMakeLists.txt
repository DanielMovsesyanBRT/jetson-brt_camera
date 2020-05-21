# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE EQUAL Debug)
  set(CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c++ Debug flags" )
  set(CMAKE_C_FLAGS_DEBUG       "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c Debug flags" )
endif()

project(video_cuda_enc C CXX CUDA)
find_package(X11 REQUIRED)

if(DEFINED SYSROOT)
  find_library(v4l2  v4l2
              PATHS ${SYSROOT}/lib ${SYSROOT}/lib/aarch64-linux-gnu )

  find_library(nvbuf_utils  nvbuf_utils
              PATHS 
                ${SYSROOT}/lib 
                ${SYSROOT}/lib/tegra
                ${SYSROOT}/lib/aarch64-linux-gnu 
                ${SYSROOT}/lib/aarch64-linux-gnu/tegra 
  )

  find_library(nvosd  nvosd
              PATHS 
                ${SYSROOT}/lib 
                ${SYSROOT}/lib/tegra
                ${SYSROOT}/lib/aarch64-linux-gnu 
                ${SYSROOT}/lib/aarch64-linux-gnu/tegra 
  )

else()
  set(v4l2 "v4l2")
  set(nvbuf_utils "nvbuf_utils")
  set(nvosd "nvosd")
endif()

file(GLOB classes
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "${CMAKE_CURRENT_SOURCE_DIR}/common/classes/*.cpp"
)

set(SOURCES 
      ${classes}
      src/video_cuda_enc.cpp
      common/algorithm/cuda/NvAnalysis.cu
      common/algorithm/cuda/NvCudaProc.cu
)
  
  
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME} 
                            PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include
                            PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/src
                             ${CMAKE_CURRENT_SOURCE_DIR}/common/algorithm/cuda
                             ${CMAKE_CURRENT_SOURCE_DIR}/common/classes
                             ${X11_X11_INCLUDE_PATH}
                            )

target_link_libraries(${PROJECT_NAME} pthread ${v4l2} EGL GLESv2 X11 ${nvbuf_utils} ${nvosd} cuda cudart)

