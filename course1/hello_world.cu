#include <cuda_runtime.h>
#include <iostream>

//__global__ void hello_world(void) {
//    printf("thread idx: %d\n", threadIdx.x);
//    if (threadIdx.x == 0) {
//        printf("GPU: Hello world!\n");
//    }
//}
//
//int main(int argc, char **argv) {
//    printf("CPU: Hello world!\n");
//    hello_world<<<1, 10>>>();   //这里相当于fastapi的异步吧
//    cudaDeviceSynchronize();    //这里相当于 阻塞，等 GPU 上的所有任务执行完
//    if (cudaGetLastError() != cudaSuccess) {
//        std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError())
//                  << std::endl;
//        return 1;
//    } else {
//        std::cout << "GPU: Hello world finished!" << std::endl;
//    }
//    std::cout << "CPU: Hello world finished!" << std::endl;
//    return 0;
//}
__global__ void hello_world(void){
    int a = 10;
    printf("thread idx: %d\n", threadIdx.x);
    if (threadIdx.x == 0) {
        printf("GPU: Hello world!\n");
        a= a+1;
        printf("a的值为 %d\n",a);
    }
}
int main(int argc,char **argv){
    printf("CPU: Hello world!\n");
    hello_world<<<1,10>>>();
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError())<<std::endl;
    }
    else{
        std::cout<<"GPU: H3333e111llo world finished!"<<std::endl;

    }
    return 0;
}


























