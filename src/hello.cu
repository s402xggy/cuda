#include <iostream>
#include <stdio.h>
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
*/

void __global__ hello_from_gpu(){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // 线程的唯一标识
    int id = threadIdx.x + blockIdx.x * blockDim.x; // 线程块中的线程id + 线程块id * grid中线程块的数量。
    printf("bid = %d, tid = %d, threadid = %d \n",bid, tid, id);
    
}

void __global__ add(int x, int y) {
    int res = x + y;
}

int main() {
    // 1、主机代码
    hello_from_gpu<<<2,16>>>(); // 设定设备的线程模型， <<<1,1>>> 第一个1为grid块gridDim.x， 第二个为线程块中的线程块(线程数量)blickDim.x。
    // 2、核函数的调用
    // 等待gpu设备执行完毕。
    cudaDeviceSynchronize();
    // 3、主机代码

    return 0;
}