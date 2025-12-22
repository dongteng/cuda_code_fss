#include <cuda_runtime.h>

#include <iostream>

__global__ void test_shuf_broadcast(int *dOutput, const int *dInput,
                                    const int srcLane) {
    int val = dInput[threadIdx.x];

    //0xFFFFFFFF 线程掩码 指定warp中哪些线程参与shuffle。 0xFFFFFFFF 表示 32 位全 1，即 warp 中所有线程都参与。
    //当前线程手里原本的值（在寄存器里）。
    //srcLane 要读取数据的 源线程的 lane ID（0~31）。所有线程会去读这个 srcLane 对应线程寄存器里的 val。
    //32表示width，表示 warp 中参与的线程数量，这里是标准 warp 大小 32。
    val = __shfl_sync(0xFFFFFFFF, val, srcLane, 32);
    dOutput[threadIdx.x] = val;
}

int main() {
    const int numThreads = 32;
    const int srcLane = 2;

    // Host arrays
    int hInput[numThreads];
    int hOutput[numThreads];

    for (int i = 0; i < numThreads; ++i) {
        hInput[i] = i;
    }

    int *dInput, *dOutput;

    cudaMalloc(&dInput, numThreads * sizeof(int));
    cudaMalloc(&dOutput, numThreads * sizeof(int));

    cudaMemcpy(dInput, hInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);

    test_shuf_broadcast<<<1, numThreads>>>(dOutput, dInput, srcLane);

    cudaMemcpy(hOutput, dOutput, numThreads * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::cout << "Broadcasting value from thread " << srcLane << ":\n";
    for (int i = 0; i < numThreads; ++i) {
        std::cout << "hOutput[" << i << "] = " << hOutput[i] << "\n";
    }

    cudaFree(dInput);
    cudaFree(dOutput);

    return 0;
}
