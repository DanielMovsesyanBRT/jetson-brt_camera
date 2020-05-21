# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

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
set(CMAKE_SKIP_BUILD_RPATH TRUE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/assets)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/assets)

add_subdirectory(script)
add_subdirectory(brt_common)
add_subdirectory(cuda_lib)
add_subdirectory(fltk_interface)
add_subdirectory(encoder)

#
# ---------- Define Project
#
project(jetson-brt_camera C CXX CUDA)

# find_package(X11 REQUIRED)

find_library(PNG
        NAMES png16
        PATHS "png/lib")

find_library(LIBZ
        NAMES z
        PATHS "png/lib")        

# find_library(VCE
#         NAMES video_cuda_enc
#         PATHS libs)


set(SOURCES 

  source/device/camera.cpp
  source/device/deserializer.cpp
  source/device/device_action.cpp
  source/device/device_manager.cpp

  source/image/isp_manager.cpp
  source/image/isp.cpp
  source/image/encoder.cpp
  source/image/image_net.cpp
  source/image/image_net_interface.cpp
  
  source/window/camera_window.cpp
  source/window/window.cpp
  source/window/window_manager.cpp
)

list(APPEND SOURCES  source/main.cpp)
        

include_directories("fltk_interface")

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_definitions(${PROJECT_NAME} PUBLIC "ARM=${ARCH_ARM}" "X86_64=${ARCH_X86_64}" "ARCH=${ARCH}")
target_include_directories(${PROJECT_NAME} 
                              PRIVATE 
                              "source" 
                              "source/device" 
                              "source/image" 
                              "source/window"
                              "png/include"
                              ${X11_X11_INCLUDE_PATH})
                              
target_link_libraries(${PROJECT_NAME} PUBLIC script cuda_lib pthread brt_common video_cuda_enc ${PNG} ${LIBZ} dl) # ${VCE})
target_compile_definitions(${PROJECT_NAME} PUBLIC HARDWARE)


set (source "${CMAKE_SOURCE_DIR}/libs/*")
set (destination "${CMAKE_SOURCE_DIR}/assets")
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source} ${destination} 
        DEPENDS ${destination}
)