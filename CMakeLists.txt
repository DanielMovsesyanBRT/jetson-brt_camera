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

add_subdirectory(script)

#
# ---------- Define Project
#
project(jetson-brt_camera C CXX)

#if (NOT FLTK_PATH)
#  if (DEFINED ENV{FLTK_PATH})
#    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#      set(FLTK_DIR "$ENV{FLTK_PATH}/Debug/share/fltk")
#    else()
#      set(FLTK_DIR "$ENV{FLTK_PATH}/Release/share/fltk")
#    endif()
#  else()
#    message(FATAL_ERROR "FLTK path is missing")
#  endif()
#endif()
#
#find_package(FLTK REQUIRED NO_MODULE)

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)

#if (DEFINED OPENCV_PATH)
#  find_package(OpenCV REQUIRED
#        PATHS ${OPENCV_PATH}
#        NO_DEFAULT_PATH)
#        
#elseif(DEFINED ENV{OPENCV_PATH})
#  find_package(OpenCV REQUIRED
#        PATHS $ENV{OPENCV_PATH}
#        NO_DEFAULT_PATH)
#else()
#  find_package(OpenCV REQUIRED)
#endif()

find_package(X11_user REQUIRED)

find_library(GL REQUIRED
        NAMES GL
        PATHS 
          ${LD_PATH}
          ${LD_PATH_EXTRA}
          )

set(SOURCES 
  source/cuda_debayering/src/cuda_debayering.cu
 
  source/device/Camera.cpp
  source/device/Deserializer.cpp
  source/device/DeviceManager.cpp
  source/device/ScriptAction.cpp
  source/device/ScriptFile.cpp
  
  # source/fltk/fltk_manager.cpp  
  
  source/image/image_processor.cpp
  source/image/image_processor.cu
  source/image/Image.cpp
  
  source/window/camera_window.cpp
  source/window/window.cpp
  source/window/window_manager.cpp
  
  source/Metadata.cpp
  source/MetaImpl.cpp
  source/Utils.cpp
)


if(EXECUTABLE)
  list(APPEND SOURCES 
        source/main.cpp
        )
  message(STATUS "Compiling Executable version")
else()
  message(STATUS "Compiling Library version")
endif()

include_directories( "source/cuda_debayering/include" "source" "source/image")
# cuda_include_directories( "source/cuda" )

if(EXECUTABLE)
  cuda_add_executable(${PROJECT_NAME} ${SOURCES})
else()
  add_library(${PROJECT_NAME} SHARED ${SOURCES})
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC "ARM=${ARCH_ARM}" "X86_64=${ARCH_X86_64}" "ARCH=${ARCH}")
target_include_directories(${PROJECT_NAME} 
                              PRIVATE 
                              "source" 
                              "source/device" 
                              # "source/fltk" 
                              "source/image" 
                              "source/window"
                              "script"
                              # ${X11_X11_INCLUDE_PATH}
                              # ${FLTK_INCLUDE_DIRS}
                              )
                              
# target_link_libraries(${PROJECT_NAME} PUBLIC script pthread ${OpenCV_LIBS} ${X11_X11_LIB} ${GL} fltk dl)
target_link_libraries(${PROJECT_NAME} PUBLIC script pthread ${OpenCV_LIBS} ${X11_X11_LIB} ${GL})

#if (ARCH EQUAL ARCH_ARM)
target_compile_definitions(${PROJECT_NAME} PUBLIC HARDWARE)
#  target_link_libraries(${PROJECT_NAME} ${NVMEDIA} ${NVMEDIA_ISC})
#endif()

#install(TARGETS ${PROJECT_NAME} DESTINATION lib)
#install(FILES JupiterIPP.hpp include/Metadata.hpp DESTINATION include)

