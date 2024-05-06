#include <stdio.h>
#include <iostream>
#include "common.cuh"


/*
    全局内存：  host和device 中的线程均可见。
        动态初始化： 使用cudaMalloc创建， cudaFree释放。
        静态全局内存：使用__device__关键字声明全局内存。


*/

__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void kernel(void) {
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(int argc, char **argv) {
    int device_id = 0;
    cudaDeviceProp device_props;
    ErrorCheck(cudaGetDeviceProperties(&device_props, device_id), __FILE__, __LINE__);
    std::cout << "运行gpu设备: " << device_props.name << std::endl;
    int h_y[2] = {10, 20};
    ErrorCheck(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2), __FILE__, __LINE__);

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    ErrorCheck( cudaDeviceSynchronize(), __FILE__, __LINE__);
    ErrorCheck( cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);

    ErrorCheck(cudaDeviceReset());

    

}