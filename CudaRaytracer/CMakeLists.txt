cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
endif()

# Configure project

project(CudaRaytracer VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(PROJECT_DESCRIPTION "Raytracer in pure cuda")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project options

# Find CUDA
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(cudaraytracer
    ${CMAKE_CURRENT_LIST_DIR}/Camera.cu
    ${CMAKE_CURRENT_LIST_DIR}/kernel.cu
    ${CMAKE_CURRENT_LIST_DIR}/raycast.cu
    ${CMAKE_CURRENT_LIST_DIR}/Scene.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MeshPrimitive.cpp)

target_link_libraries(cudaraytracer PUBLIC
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )


target_include_directories(cudaraytracer
    PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include(GNUInstallDirs)

install(
    TARGETS cudaraytracer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

