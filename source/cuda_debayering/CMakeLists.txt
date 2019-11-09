cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda_debayering LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} SHARED src/cuda_debayering.cu)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBS})

add_executable(test_cuda_debayering test/test_cuda_debayering.cpp)
target_link_libraries(test_cuda_debayering
                      ${PROJECT_NAME})
