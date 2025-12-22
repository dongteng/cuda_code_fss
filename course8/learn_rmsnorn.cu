#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void row_rmsnorm_f32_dim_cpu(float *in, float *weight, float *out, int batch,int size, float eps) {
    //写这个算子 同时也熟悉了大模型的RMS组件
    //已知矩阵规模为batch * size
    for(int i=0; i<batch; ++i) {

        //挪到第i个循环的输入输出
        float *in_ptr = in + i * size;
        float *out_ptr = out + i * size;

        //求每个循环的一个平方
        float sum = 0.0f;
        for(int j=0; j<size; ++j) {
            float val = in_ptr[j];
            sum += val * val;
        }
        //求每个循环的平方根
//        float rms = 1.0f / std::sqrt(sum / size + eps);
        float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);
        for(int j=0; j<size; ++j) {
            float x = in_ptr[j] * weight[j];
            out_ptr[j] = x * rms;
        }

    }
}

__inline__ __device__ float block_reduce(float val){
    //注意此处命名，是块规约，而原语只有在warp内
    //有多个warp, 每个warp做一个规约
    const int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;  //线程在warp内的编号0-31
    int warp_id = tid / warpSize; //线程所在的warp编号（一个 block 可能有多个 warp）

    //warp内规约，每个warp的值在 每个warp的thread0那里
    for(int offset = warpSize/2 ; offset > 0 ; offset /= 2){
        //这里的0xFFFFFFFF表示所有线程都参与
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float warpSums[32];  // 每个warp一个值 哎？这里不应该是线程块个数么？
    if(lane == 0){
        warpSums[warp_id] = val;    // 每个线程
    }
    __syncthreads();

    //因为函数名是线程块规约，所以还需要第二阶段规约
    //现在最终值都在共享内存里边了 一共有batch个，在这里是32个、
    //最终归约，只有warp0的线程参与
    if(warp_id == 0){
        //重新给参与计算的 32个线程赋值，值为共享内存的，以便再用一次__shfl_down_sync
//        val = warpSums[tid];//这种不对
        val =(tid < (blockDim.x+ warpSize-1)/warpSize) ? warpSums[tid] : 0.0f; //这个判断是让有意义的数值参与规约
        __syncthreads(); //为什么正确答案里没有这个同步？
        for(int offset = warpSize/2 ; offset > 0 ; offset /= 2){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        }
    else{
        val = 0.0f;
    }
    return val;
}

__global__ void row_rmsnorm_f32_dim(float* in, float*wei , float *out, int batch ,int size, float eps ){
    const int bid = blockIdx.x; //每个快负责一个batch
    if(bid>batch) return;

    float *block_in = in + bid*size;
    float *block_out = out + bid*size;
    float sum = 0.0f;

    //每个线程块算一行， 每个线程负责 32个， 利用warp 规约、
    //首先要注意的是 每个线程负责几个数值，进规约之前，先把这个值算出来。 比如线程0 负责0 . 1024号 2048号数据
    for(int i = threadIdx.x; i<size ; i+=blockDim.x){
//        sum += block_in[i] * block_in[i]; //这个写法就没有下一个写法好呢！
        float x = block_in[i] ;
        sum += x * x;
    }

    //每个块在共享内存声明一个值 用于接收warp规约的结果  此处存疑，规约是warp内发生的，有多个warp哦
    __shared__ float shared_val;
    sum = block_reduce(sum);
    if(threadIdx.x == 0){
        //最终结果在0号warp的0号threadid  那必然总的id是0
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    const float scale = rsqrtf(sum/static_cast<float>(size) + eps);

    //最后结果汇入 block_out
    //已知总和了，那就每个元素除以总和即可。 一个thread可能处理多个数据 此时写回结果该如何处理呢？
    //哦 自己确实没想到 呃
    for(int i = threadIdx.x ; i< size; i+=blockDim.x){
        float x = block_in[i] * wei[i];
        block_out[i] = x * scale;
    }

}

__global__ void rmsnorm_fp32_dim_simd(float *in, float *weight, float *out, int batch, int size, float eps) {
    //向量化执行rmsnorm
    //一个线程块负责一行
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if(bid > batch) return;
    float *block_in = in + bid * size;
    float *block_out = out + bid * size; //挪动位置

    constexpr int pack_size =4;
    const int pack_num = size / pack_size; //一共多少个完整的float4
    const int pack_off = size % pack_size; //最后剩余多少个数据

    float sum = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(block_in);

    for(int i = tid; i<pack_num; i+=blockDim.x){
        //每个线程可能负责多个数据
        float4 in_float4 = in_pack[i];
//        float4 in_float4 = *(in_pack+i);
        sum += in_float4.x * in_float4.x + in_float4.y * in_float4.y + in_float4.z * in_float4.z + in_float4.w * in_float4.w;
    }
    //计算剩余不足4个的
    //起始感觉最后一项没必要 甚至+1都行，因为最后剩3个数据最多。加blockDim.x肯定超范围了
    for(int i = tid+pack_num*pack_size; i<size; i+=blockDim.x){
        sum += block_in[i]*block_in[i];
    }

    __shared__ float shared_val;
    sum = block_reduce(sum);
    if(threadIdx.x == 0){
        shared_val = sum;
    }
    
}



int main(){
    const int batch = 16;
    const int size = 1024;
    const float eps = 1e-6f;
    const int total = batch *size;

    std::vector<float> h_input(total);
    std::vector<float> h_weight(size);
    std::vector<float> h_output_cpu(total);
    std::vector<float> h_output_cuda(total);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < total; ++i) {
        h_input[i] = dis(gen);
    }
    for (int i = 0; i < size; ++i) {
        h_weight[i] = dis(gen);
    }

    //cpu version
    auto start = std::chrono::high_resolution_clock::now();
    row_rmsnorm_f32_dim_cpu(h_input.data(), h_weight.data(), h_output_cpu.data(), batch, size, eps);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU RMSNorm took " << duration.count() << " microseconds.\n";


    //cuda setup
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    const int block_size =1024;
    const int grid_size = batch;
    dim3 grid(grid_size);
    dim3 block(block_size);
}