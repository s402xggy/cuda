#include <iostream>
#include <stdio.h>
#include "common.cuh"
/*
    核函数：
        1、核函数在gpu上进行并行执行。
        2、限定词__global__修饰
        3、返回值为void
    注意事项：
        1、核函数只能访问gpu内存
        2、核函数不能使用变长参数
        3、核函数B不能使用静态变量。
        4、核函数不能使用函数指针
        5、核函数具有异步性

    设备函数：
        1、定义只能执行在gpu设备上的函数。
        2、设备函数只能被核函数或其他设备函数调用。
        3、设备函数用__device__修饰。
    核函数：
        1、一般由主机调用，在设备中执行。
        2、__global__ 修饰符既不能和__host__同时使用， 也不能和__device__同时使用。
    主机函数：
        1、主机端普通c++函数可以使用__host__修饰，也可以省略。
        2、可以用__host__ 和 __device__ 同时修饰一个函数减少冗余代码。
*/

// 设备函数
__device__ int Add(int x, int y) {
    return x + y;
}

__global__ void AddFromGpu(float *a, float *b, float *c, const int n) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;
    if (id < n)
        c[id] = Add(a[id], b[id]);
    else 
        return ;
}


int main() {
    SetGpu();

    int elem_count = 512;
    size_t bytes_count = elem_count * sizeof(float);

    // 1、开辟主机中的内存, 并初始化为0；
    float *host_aptr, *host_bptr, *host_cptr;
    host_aptr = new float[elem_count];
    host_bptr = new float[elem_count];
    host_cptr = new float[elem_count];
    if (host_aptr != nullptr && host_bptr != nullptr && host_cptr != nullptr) {
        memset(host_aptr, 0, bytes_count);
        memset(host_bptr, 0, bytes_count);
        memset(host_cptr, 0, bytes_count);
    }
    // 2、分配设备内存。
    float *device_aptr, *device_bptr, *device_cptr;
    cudaMalloc((void **)&device_aptr, bytes_count);
    cudaMalloc((void **)&device_bptr, bytes_count);
    cudaMalloc((void **)&device_cptr, bytes_count);
    if (device_aptr != nullptr && device_bptr != nullptr && device_cptr != nullptr) {
        cudaMemset(device_aptr, 0, bytes_count);
        cudaMemset(device_bptr, 0, bytes_count);
        cudaMemset(device_cptr, 0, bytes_count);
    }

    if (host_aptr != nullptr && host_bptr != nullptr && host_cptr != nullptr) {
        memset(host_aptr, 0, bytes_count);
        memset(host_bptr, 0, bytes_count);
        memset(host_cptr, 0, bytes_count);
    }
    // 3、初始化数据
    srand(666);
    InitalData(host_aptr, elem_count);
    InitalData(host_bptr, elem_count);

    // 4、数据从主机复制到设备
    cudaMemcpy(device_aptr, host_aptr, bytes_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_bptr, host_bptr, bytes_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_cptr, host_cptr, bytes_count, cudaMemcpyHostToDevice);

    // 5、调用核函数准备计算
    dim3 block(32);
    dim3 grid(elem_count / 32);
    AddFromGpu<<<grid, block>>>(device_aptr, device_bptr, device_cptr, elem_count);
    
    cudaDeviceSynchronize();

    // 6、将计算得到的数据从设备传给主机
    cudaMemcpy(host_cptr, device_cptr, bytes_count, cudaMemcpyDeviceToHost);

    for (int i = 0; i < elem_count; i++) {
        printf("idx = %2d\tmatrix_a:%.2f\tmatrix_b:%.2f\tresult=%.2f\n", i+1, host_aptr[i], host_bptr[i], host_cptr[i]);
    }
    // 释放内存
    delete host_aptr;
    delete host_bptr;
    delete host_cptr;
    cudaFree(device_aptr);
    cudaFree(device_bptr);
    cudaFree(device_cptr);

    cudaDeviceReset();

    return 0;
}