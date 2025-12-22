#include <cuda_runtime.h>

#include <chrono>  // 用于 CPU 计时
#include <iostream>
#include <numeric>
#include <vector>
const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;

float reduce_cpu(const std::vector<float>& data) {
    float sum = 0.0f;
    for(float val:data){
        sum += val;
    }
    return sum;
}

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

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce_v1(float *g_idata, float *g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;


    //这里怎么都不明白 为啥先*2了 后续g_idata里边又加了blockDim.x？
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce_v2(float *g_idata, float *g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}




int main(){


    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;  // 简单起见，全部初始化为1.0
    }

    //cpu 计时开始
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = reduce_cpu(h_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU result: " << cpu_duration.count() << std::endl;


    float *d_data, *d_result, *d_final_result;
    float gpu_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, num_blocks * sizeof(float));
    cudaMalloc(&d_final_result, 1 * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), N * sizeof(float),cudaMemcpyHostToDevice);


    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    reduce_v0<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    reduce_v0<<<1, num_blocks>>>(d_result, d_final_result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_v0 = 0;
    cudaEventElapsedTime(&gpu_time_v0, start, stop);


    std::cout << "GPU v0 kernel time: " << gpu_time_v0 << " ms" << std::endl;
    cudaMemcpy(&gpu_result, d_final_result, sizeof(float),cudaMemcpyDeviceToHost);
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