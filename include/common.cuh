#ifndef __COMMON_CUH__
#define __COMMON_CUH__

// 错误检查函数
// filename 一般使用__FILE__ 参数lineNumber 一般使用 __LINE__
cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber) {
    if (error_code != cudaSuccess) {
        printf("CUDA error:\r\ncode = %d, name = %s ,description = %s\r\nfile = %s, line = %d\r\n",error_code, 
        cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

void SetGpu(){
    // 1、检测gpu数量
    int id_device_count = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&id_device_count), __FILE__, __LINE__);
    if (error != cudaSuccess || id_device_count == 0) {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else {
        printf("The count of GPU is %d.\n", id_device_count);
    }
    // 2、设置执行GPU 为 0;
    int idev = 0;
    error = cudaSetDevice(idev);
    if (error != cudaSuccess ) {
        printf("fail to set GPU 0 for computing.!\n");
        exit(-1);
    }
    else {
        printf("set GPU 0 for conputing.\n");
    }
}

void InitalData(float *addr, int elem_count) {
    for (int i = 0; i < elem_count; i++) {
        addr[i] = (float)(rand() & 0xff) / 10.f;
    }
}








#endif
