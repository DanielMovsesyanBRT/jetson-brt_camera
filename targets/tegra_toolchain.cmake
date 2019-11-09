# Copyright (c) 2019, Blue River

set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED TOOLCHAIN_PREFIX)
  if (DEFINED ENV{TOOLCHAIN_PREFIX})
    set(TOOLCHAIN_PREFIX $ENV{TOOLCHAIN_PREFIX} CACHE STRING "Tool-chain prefix" FORCE)
  else()
    set(TOOLCHAIN_PREFIX "${CMAKE_SYSTEM_PROCESSOR}-gnu-linux")
  endif()
endif()


if(NOT DEFINED TOOLCHAIN_PATH)
  if(DEFINED ENV{TOOLCHAIN_PATH})
    message(STATUS "TOOLCHAIN_PATH = ENV : $ENV{TOOLCHAIN_PATH}")
    set(TOOLCHAIN_PATH $ENV{TOOLCHAIN_PATH} CACHE STRING "Path to the toolchain path for cross-compilation" FORCE)
  else()
    message(FATAL_ERROR "TOOLCHAIN_PATH Variable has to be specified")
  endif()
elseif(NOT DEFINED ENV{TOOLCHAIN_PATH})
  set(ENV{TOOLCHAIN_PATH} ${TOOLCHAIN_PATH})
endif()

set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-g++")
set(CMAKE_C_COMPILER "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-gcc")

set(CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG       "-Wl,-rpath-link,")       # -rpath
set(CMAKE_EXECUTABLE_RUNTIME_C_FLAG         "-Wl,-rpath-link,")       # -rpath
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG     "-Wl,-rpath-link,")
set(CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG   "-Wl,-rpath-link,")


# setup compiler for cross-compilation
set(CMAKE_CXX_FLAGS           "-fPIC"               CACHE STRING "c++ flags")
set(CMAKE_C_FLAGS             "-fPIC"               CACHE STRING "c flags")
set(CMAKE_SHARED_LINKER_FLAGS ""                    CACHE STRING "shared linker flags")
set(CMAKE_MODULE_LINKER_FLAGS ""                    CACHE STRING "module linker flags")
set(CMAKE_EXE_LINKER_FLAGS    ""                    CACHE STRING "executable linker flags")

if(NOT DEFINED ROOT_FS)
  if(DEFINED ENV{ROOT_FS})
    message(STATUS "ROOT_FS = ENV : $ENV{ROOT_FS}")
    set(ROOT_FS $ENV{ROOT_FS} CACHE STRING "Path to the toolchain path for cross-compilation" FORCE)
  else()
    message(FATAL_ERROR "NVidia Path is not defined through NVIDIA_DIR variable")
  endif()
elseif(NOT DEFINED ENV{ROOT_FS})
  set(ENV{ROOT_FS} ${ROOT_FS})
endif()

#if(NOT DEFINED EXTRA_INCLUDE)
#  if(DEFINED ENV{EXTRA_INCLUDE})
#    message(STATUS "EXTRA_INCLUDE = ENV : $ENV{EXTRA_INCLUDE}")
#    set(EXTRA_INCLUDE $ENV{EXTRA_INCLUDE} CACHE STRING "Path to the toolchain path for cross-compilation" FORCE)
#  else()
#    message(FATAL_ERROR "NVidia Path is not defined through NVIDIA_DIR variable")
#  endif()
#elseif(NOT DEFINED ENV{EXTRA_INCLUDE})
#  set(ENV{EXTRA_INCLUDE} ${EXTRA_INCLUDE})
#endif()
#

set(LD_PATH ${ROOT_FS}/lib)
set(LD_PATH_EXTRA ${LD_PATH}/aarch64-linux-gnu)
set(INCLUDE_PATH ${ROOT_FS}/include)

set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} -L${LD_PATH} -L${LD_PATH_EXTRA} -Wl,--stats -Wl,-rpath-link,${LD_PATH} -Wl,-rpath-link,${LD_PATH_EXTRA}")
set(CMAKE_MODULE_LINKER_FLAGS   "${CMAKE_MODULE_LINKER_FLAGS} -L${LD_PATH} -L${LD_PATH_EXTRA} -Wl,-rpath-link,${LD_PATH} -Wl,-rpath-link,${LD_PATH_EXTRA}")
set(CMAKE_EXE_LINKER_FLAGS      "${CMAKE_EXE_LINKER_FLAGS} -L${LD_PATH} -L${LD_PATH_EXTRA} -Wl,-rpath-link,${LD_PATH} -Wl,-rpath-link,${LD_PATH_EXTRA}")
 
set(X11_INC_SEARCH_PATH ${INCLUDE_PATH})
set(X11_LIB_SEARCH_PATH ${LD_PATH} ${LD_PATH_EXTRA}) # /aarch64-linux-gnu)

# Setting VIBRANTE definitions for Driveworks
#add_definitions(-DVIBRANTE)


# These two flags are necessary for cmake version 3.14.2
# For some reason it will exclude NVIDIA directories from the flags.make file if they are not
# listed in these two lists...
list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${INCLUDE_PATH})
list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES ${INCLUDE_PATH})

include_directories(BEFORE SYSTEM ${INCLUDE_PATH})
