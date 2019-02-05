###############################################################################
# COMPILER OPTIONS
###############################################################################

set(ROOT_PATH		${PROJECT_SOURCE_DIR}/..)
set(PROJECT_COMMON  ${ROOT_PATH}/common)
set(OPENCV   		/home/util/opencv)

set(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 ")

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} arch x86_64")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} arch x86_64")
else(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

if(NOT CMAKE_BUILD_TYPE)
# DO NOT USE Release mode as the -O3 compile option will cause issues 
# against CUDA, NVENC, ...
    set(CMAKE_BUILD_TYPE Product)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL Product)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 ")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ")
else(${CMAKE_BUILD_TYPE} STREQUAL Debug)
	set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -g -DDEBUG")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DDEBUG")
endif(${CMAKE_BUILD_TYPE} STREQUAL Product)

OPTION(DEFINE_PROFILE	"Build the project using profiling code"	OFF)
IF(DEFINE_PROFILE)
	MESSAGE("Adding Profile flag...")
	ADD_DEFINITIONS(-DPROFILE)
ENDIF(DEFINE_PROFILE)

FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_HOST_COMPILER	${CMAKE_CXX_COMPILER})
set(CUDA_TOOLKIT  	/usr/local/cuda)
set(CUDA_LIBS		${CUDA_TOOLKIT}/lib64)
set(NVENC_LIBS		/usr/lib/nvidia-396 /usr/lib/nvidia-410 )
set(CUDA_INCS		${CUDA_TOOLKIT}/include)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2 -gencode arch=compute_50,code=sm_50)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(OPENCV_INC   	/home/util/opencv/include)
set(OPENCV_LIB		/home/util/opencv/lib)