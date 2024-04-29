cmake_minimum_required(VERSION 3.24)
project(cuda LANGUAGES CXX CUDA)


set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++17 -O0 -g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_87,code=compute_87
)
set(Common_Inc_Dir ${PROJECT_SOURCE_DIR}/include)

add_executable(set_device src/set_device.cu)
target_include_directories(set_device PRIVATE ${Common_Inc_Dir})

add_executable(error_check src/error_check.cu)
target_include_directories(error_check PRIVATE ${Common_Inc_Dir})