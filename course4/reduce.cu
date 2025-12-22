#include <cuda_runtime.h>

#include <chrono>  // 用于 CPU 计时
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;  // 1M elements


//这段 CUDA kernel 代码的作用是做 block内的归约求和（reduction），即每个线程块把它负责的一段数据加起来，最后把结果存入 g_odata。
__global__ void reduce_v0(float *g_idata, float *g_odata) {
    __shared__ float sdata[BLOCK_SIZE];// block 内共享内存

    unsigned int tid = threadIdx.x; //在线程块内的索引
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//全局索引

    //加载数据到共享内存
    if (i < N) {  // 防止越界访问
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    //这段是二叉树归约，每轮迭代把数据规模减半。以 Block 0（sdata=[1,2,3,4,5,6,7,8]）为例：第一轮 s=1
    //tid=0：sdata[0] += sdata[1] → 1+2=3
    //tid=2：sdata[2] += sdata[3] → 3+4=7
    //tid=4：sdata[4] += sdata[5] → 5+6=11
    //tid=6：sdata[6] += sdata[7] → 7+8=15
    //结果： [3,2,7,4,11,6,15,8]
    //第二轮 s=2
    //tid=0：sdata[0] += sdata[2] → 3+7=10
    //tid=4：sdata[4] += sdata[6] → 11+15=26
    //结果： [10,2,7,4,26,6,15,8]
    //第三轮 s=4
    //tid=0：sdata[0] += sdata[4] → 10+26=36
    //结果： [36,2,7,4,26,6,15,8]
    //最终 sdata[0] = 36 是 block 0 负责的总和。
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// CPU验证函数
float reduce_cpu(const std::vector<float> &data) {
    float sum = 0.0f;
    for (float val: data) {
        sum += val;
    }
    return sum;
}

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<float> h_data(N);

    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;  // 简单起见，全部初始化为1.0
    }

    // -------------------------------
    // CPU 计时开始
    auto cpu_start = std::chrono::high_resolution_clock::now();

    float cpu_result = reduce_cpu(h_data);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    // CPU 计时结束
    // -------------------------------

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    float *d_data, *d_result;
    float *d_final_result;
    float gpu_result;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, num_blocks * sizeof(float));
    cudaMalloc(&d_final_result, 1 * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // -------------------------------
    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce_v0<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    reduce_v0<<<1, num_blocks>>>(d_result, d_final_result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束
    // -------------------------------

    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&gpu_result, d_final_result, sizeof(float),
               cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << gpu_result << std::endl;

    if (abs(cpu_result - gpu_result) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    } else {
        std::cout << "Result verification failed!" << std::endl;
    }

    // 清理资源
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_final_result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
