#
# Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
# All content herein is protected by U.S. copyright and other applicable intellectual property laws
# and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
# Reuse of any of the content for any purpose without the permission of Keith Ha
# is strictly and expressively prohibited.
#

cmake_minimum_required(VERSION 3.1.0)
project(meanfilter)

include(build.cmake)
include(env.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -rdc=true
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_75,code=sm_75
        )

include_directories(
        ${APP_INC_PATH}
        ${CUDA_INCS}
        ${PROJECT_BINARY_DIR}
)

link_directories(
        ${CMAKE_SOURCE_DIR}
        ${CUDA_LIBS}
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
)

file( GLOB CUDA_SRC_FILES kernel.cu)

set(MEANFILTER_FILES
        ThreadManager.cpp
        meanfilter.cpp
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(DYNAMIC_LIBRARIES
            libm.so
            libz.so
            libdl.so
            libpthread.so
            libcuda.so
            libcudart.so
        )
    set(STATIC_LIBRARIES
            libcudadevrt.a
        )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)

cuda_add_executable(meanfilter ${MEANFILTER_FILES} ${CUDA_SRC_FILES})
#add_executable(meanfilter ${MEANFILTER_FILES})

SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

target_link_libraries(meanfilter ${STATIC_LIBRARIES} ${DYNAMIC_LIBRARIES} )