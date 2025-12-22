// rmsnorm_cuda_test.cpp
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void row_rmsnorm_f32_dim_cpu(float *in, float *weight, float *out, int batch,
                             int size, float eps) {
    for (int i = 0; i < batch; ++i) {
        float *in_ptr = in + i * size;
        float *out_ptr = out + i * size;

        float sum = 0.0f;
        for (int j = 0; j < size; ++j) {
            float val = in_ptr[j];
            sum += val * val;
        }
        float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);

        for (int j = 0; j < size; ++j) {
            float x = in_ptr[j] * weight[j];
            out_ptr[j] = x * rms;
        }
    }
}

//inline内联函数 编译器尽量把函数体直接展开到调用处，而不是生成单独函数调用
//一个块级规约函数 用于把块内所有的val求和，最终返回每个线程块的总和（通常只有thread0最终用结果）
//每个线程都会调用这个函数 把自己的val传进去
__inline__ __device__ float block_reduce(float val) {
    const int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;  //线程在warp内的编号0-31
    int warp_id = tid / warpSize; //线程所在的warp编号（一个 block 可能有多个 warp）

    // Warp-level reduction 第一阶段 Warp内规约  0xFFFFFFFF：表示所有线程都参与
    //逐循环展开：
    //offset = 16：lane 0 加 lane 16 的值，lane 1 加 lane 17 ...
    //offset = 8：lane 0 加 lane 8 的值 ...
    //offset = 4、offset = 2、offset = 1
    //最终 lane = 0 的线程保存 warp 的和
    //在 C/C++ 里，如果 for/while/if 后面只有一条语句，可以省略大括号。
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        //这里为什么显式传val?
        //当前线程（lane i）读取 lane (i+offset) 的值，并返回它。
        //同时，硬件会确保 lane i 的 val 也可以被其他线程（lane i-offset）读取，但这是硬件内部实现的广播，不是第二次操作。
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // Write warp result to shared memory
    //只有warp的lane 0 线程把结果写入 warpSums
    __shared__ float warpSums[32];  // Max 32 warps per block
    if (lane == 0) {
        warpSums[warp_id] = val;
    }
    __syncthreads(); //保证所有warp写完 当一个 block 的线程执行到 __syncthreads() 时，所有线程都会停下来，直到 block 内的每个线程都到达这个屏障

    // Final reduction: only first warp participates
    if (warp_id == 0) {//只有 warp_id == 0 的线程（即前 32 个线程）参与最终归约,
        // val这行代码的目的是给warp0里的32个线程重新赋值val，并且每个
        //三元运算符 如果 condition 为真，就取 expr1  否则取 expr2 ；(blockDim.x + warpSize - 1) / warpSize 是计算当前block里 warp的数量;warpSize - 1的目的是向上取整
        //blockDim.x = block 内线程总数
        //warpSize = 每个 warp 的线程数（32）
        val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            //val = val(自己原来的) + (来自 lane i+offset 的值)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    } else {
        val = 0.0f;
    }
    return val;
}

__global__ void row_rmsnorm_f32_dim_simd(float *in, float *wei, float *out,
                                         int batch, int size, float eps) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid >= batch) {
        return;
    }

    float *block_in = in + bid * size; //in out本身就是指向全局内存矩阵的指针
    float *block_out = out + bid * size;

    //SIMD打包float4  为什么 pack_size = 4？  CUDA 对齐内存时，float4（16 字节）加载效率高；为什么4行8不行？float4 是 CUDA 原生支持的最大 SIMD 向量类型
    constexpr int pack_size = 4;
    const int pack_num = size / pack_size; //有多少个完整的float
    const int pack_off = pack_size * pack_num; //打包之后的偏移 剩余的尾元素从这里开始

    float sum = 0.0f;
    float4 *in_pack = reinterpret_cast<float4 *>(block_in);
    //tid 是线程的索引
    for (int i = tid; i < pack_num; i += blockDim.x) { //这里是一个块负责一个batch的数据
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }
    // 计算剩余不足4个的
    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        sum += block_in[i] * block_in[i];
    }

    __shared__ float shared_val;
    //每个线程都把自己算的部分交给block_reduce 最终只有一个线程拿到整个Block的和
    //现在问题来了 只有thread0才知道答案 其他线程不知道
    sum = block_reduce(sum);

    if (threadIdx.x == 0) {
        shared_val = sum;//thread 0 把结果写到共享内存 shared_val（黑板）。
    }
    __syncthreads(); //大家都等一下 等thread0 把结果写完 再往下走  ，保证所有线程都能读到黑板上的数据
    sum = shared_val;//每个线程都去黑板超结果  更新到自己的sum

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    float4 *wei_pack = reinterpret_cast<float4 *>(wei);
    float4 *out_pack = reinterpret_cast<float4 *>(block_out);
    for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        *(out_pack + i) = make_float4(
                scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }
    //处理尾元素
    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        block_out[i] = wei[i] * block_in[i] * scale;
    }
}
//batch是行数
__global__ void row_rmsnorm_f32_dim(float *in, float *wei, float *out,
                                    int batch, int size, float eps) {
    const int bid = blockIdx.x;
    if (bid >= batch) return;

    float *block_in = in + bid * size;
    float *block_out = out + bid * size;
    float sum = 0.0f;


    //这一段代码表示 每个线程只处理输入样本的一部分元素，并把这部分的平方和累加到 sum 里。 此处blockDim.x为1024
    //    逐循环展开举例（假设 size = 4096，blockDim.x = 1024）：
    //    学生 0（thread 0）：处理 i = 0, 1024, 2048, 3072
    //    学生 1（thread 1）：处理 i = 1, 1025, 2049, 3073
    //    学生 2（thread 2）：处理 i = 2, 1026, ...
    //    每个学生在自己的本子（sum）上记部分平方和。
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float x = block_in[i];
        sum += x * x;
    }


    //一个块内规约操作，把所有学生的sum累加
    __shared__ float shared_val;
    sum = block_reduce(sum);

    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val; //为什么要把规约后的结果写入共享内存，再把共享内存的变量写回sum?因为每个线程自己的寄存器 sum 还保存的是旧的部分和。我们需要把 黑板上的最终结果（shared_val）抄回每个线程的 sum，这样后续计算 scale 时大家一致。

    //sum：此时每个线程的 sum 都是 整行的平方和（之前通过共享内存广播）。
    //sum / size：算的是 均方值 (Mean of squares)。
    //+ eps：为了数值稳定，防止除 0。
    //rsqrtf(x)：快速计算 1 / sqrt(x)。
    //这个都计算不浪费计算资源？
    //看起来有重复计算，但这段计算 非常轻量（1 次除法、1 次加法、1 次 rsqrtf）。
    //CUDA 的设计理念：避免引入额外的同步开销（比如让 thread 0 算，再广播）。
    //相比引入共享内存同步，这点重复计算更划算，尤其在 GPU 上。
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    //每个线程只处理 自己负责的数据片段，
    //循环过程：
    //第1轮：i = 5
    //   x = block_in[5] * wei[5]
    //   block_out[5] = x * scale
    //
    //第2轮：i = 1029
    //   x = block_in[1029] * wei[1029]
    //   block_out[1029] = x * scale
    //
    //第3轮：i = 2053
    //   x = block_in[2053] * wei[2053]
    //   block_out[2053] = x * scale
    //
    //第4轮：i = 3077
    //   x = block_in[3077] * wei[3077]
    //   block_out[3077] = x * scale
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float x = block_in[i] * wei[i];
        block_out[i] = x * scale;
    }
}

float compute_max_error(const std::vector<float> &cpu_out,
                        const std::vector<float> &cuda_out, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = std::abs(cpu_out[i] - cuda_out[i]);
        max_err = std::max(max_err, err);
        if (max_err > 1.f) {
            std::cout << "Error at index " << i << ": CPU = " << cpu_out[i]
                      << ", CUDA = " << cuda_out[i] << ", Error = " << err << "\n";
            break;
        }
    }
    return max_err;
}

// ----------------------------
// Main Function
// ----------------------------
int main() {
    const int batch = 16;
    const int size = 1024;
    const float eps = 1e-6f;
    const int total = batch * size;

    // Host memory
    std::vector<float> h_input(total);
    std::vector<float> h_weight(size);
    std::vector<float> h_output_cpu(total);
    std::vector<float> h_output_cuda(total);

    // Random init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < total; ++i) {
        h_input[i] = dis(gen);
    }
    for (int i = 0; i < size; ++i) {
        h_weight[i] = dis(gen);
    }

    // CPU version
    auto start = std::chrono::high_resolution_clock::now();
    row_rmsnorm_f32_dim_cpu(h_input.data(), h_weight.data(), h_output_cpu.data(),
                            batch, size, eps);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU RMSNorm took " << duration.count() << " microseconds.\n";

    // CUDA setup
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), total * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float),
               cudaMemcpyHostToDevice);

    // Kernel launch config
    const int block_size = 1024;
    const int grid_size = batch;  // One block per batch row
    dim3 grid(grid_size);
    dim3 block(block_size);

    // CUDA timing with events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    int warpup = 10;
    for (int i = 0; i < warpup; i++) {
        // Warm-up run
        row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
                                             size, eps);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != 0) {
        printf("cuda error:%d\n", err);
    }
    cudaEventRecord(start_event);
    // row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
    // size, eps);
    int test_iter = 10;
    for (int i = 0; i < test_iter; ++i) {
        row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
                                             size, eps);
    }
    cudaEventRecord(stop_event);

    // Wait and measure
    cudaEventSynchronize(stop_event);
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start_event, stop_event);  // ms

    // Copy result back
    cudaMemcpy(h_output_cuda.data(), d_output, total * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "CUDA RMSNorm took " << cuda_time * 1000 / test_iter
              << " microseconds.\n";

    // Compare results
    float max_error = compute_max_error(h_output_cpu, h_output_cuda, total);
    std::cout << "Max absolute error (CPU vs CUDA): " << max_error << "\n";

    // Optional: print first few values
    std::cout << "\nFirst 10 outputs (CPU vs CUDA):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "CPU: " << h_output_cpu[i] << " | CUDA: " << h_output_cuda[i]
                  << " | Diff: " << std::abs(h_output_cpu[i] - h_output_cuda[i])
                  << "\n";
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
