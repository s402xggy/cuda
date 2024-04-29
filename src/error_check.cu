#include <stdio.h>
#include "common.cuh"


int main() {
    SetGpu();
    float *host_a;
    host_a = new float(4);
    memset(host_a, 0, sizeof(float));

    float *device_a;
    cudaError_t error = ErrorCheck(cudaMalloc((float **)&device_a, 4), __FILE__, __LINE__);
    cudaMemset(device_a , 0, sizeof(float));

    // 从主机到设备拷贝。但是使用的是cudaMemcpyDeviceToHost，将会报错
    ErrorCheck(cudaMemcpy(device_a, host_a, sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    delete host_a;
    ErrorCheck(cudaFree(device_a), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);



    return 0;
}