# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#  message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE must be specified")
# endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/X11)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c++ Debug flags" )
  set(CMAKE_C_FLAGS_DEBUG       "-O0 -g -DDEBUG -D_DEBUG" CACHE STRING "c Debug flags" )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/assets)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/assets)


INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/script
  ${CMAKE_SOURCE_DIR}/brt_common
  ${CMAKE_SOURCE_DIR}/brt_common/image)

add_subdirectory(script)
add_subdirectory(brt_common)
add_subdirectory(fltk_interface)

#
# ---------- Define Project
#
project(jetson-brt_camera C CXX CUDA)

find_package(X11 REQUIRED)

find_library(PNG
        NAMES png16
        HINTS "${CMAKE_SOURCE_DIR}/png/lib")        

find_library(LIBZ
        NAMES z
        HINTS "${CMAKE_SOURCE_DIR}/png/lib")        

set(SOURCES 

  source/device/camera.cpp
  source/device/deserializer.cpp
  source/device/device_action.cpp
  source/device/device_manager.cpp

  source/image/debayer.cu
  source/image/isp_manager.cpp
  source/image/isp.cpp
  
  source/window/camera_window.cpp
  source/window/window.cpp
  source/window/window_manager.cpp
)


if(EXECUTABLE)
  list(APPEND SOURCES 
        source/main.cpp)
        
  message(STATUS "Compiling Executable version")
else()
  message(STATUS "Compiling Library version")
endif()

include_directories("brt_common")
include_directories("fltk_interface")

if(EXECUTABLE)
  add_executable(${PROJECT_NAME} ${SOURCES})
else()
  add_library(${PROJECT_NAME} SHARED ${SOURCES})
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC "ARM=${ARCH_ARM}" "X86_64=${ARCH_X86_64}" "ARCH=${ARCH}")
target_include_directories(${PROJECT_NAME} 
                              PRIVATE 
                              "source" 
                              "source/device" 
                              "source/fltk" 
                              "source/image" 
                              "source/window"
                              "script" 
                              "png/include" 
                              ${X11_X11_INCLUDE_PATH})
                              
target_link_libraries(${PROJECT_NAME} PUBLIC script pthread ${PNG} ${LIBZ})
target_compile_definitions(${PROJECT_NAME} PUBLIC HARDWARE)

