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

project(fltk_interfaces C CXX)

if (NOT FLTK_PATH)
  if (DEFINED ENV{FLTK_PATH})
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(FLTK_DIR "$ENV{FLTK_PATH}/Debug/share/fltk")
    else()
      set(FLTK_DIR "$ENV{FLTK_PATH}/Release/share/fltk")
    endif()
  else()
    message(FATAL_ERROR "FLTK path is missing")
  endif()
endif()

find_package(FLTK REQUIRED NO_MODULE)
find_package(X11_user REQUIRED)
find_library(GL REQUIRED
        NAMES GL
        PATHS 
          ${LD_PATH}
          ${LD_PATH_EXTRA})

set(SOURCES 
  fltk_manager.cpp
  camera_menu.cxx
)
  
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${X11_X11_LIB} ${GL} fltk)

target_include_directories(${PROJECT_NAME} 
                            PUBLIC 
                              ${CMAKE_CURRENT_SOURCE_DIR}
                            PRIVATE
                              ${X11_X11_INCLUDE_PATH}
                              ${FLTK_INCLUDE_DIRS})
