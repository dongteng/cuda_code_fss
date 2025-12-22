#include <chrono>   // for timing
#include <cmath>    // for INFINITY
#include <cstdlib>  // for malloc/free
#include <iostream>

// CPU implementation
void softmax_forward_cpu(float *out, const float *inp, int N, int C) { //输入是N*C的张量
    for (int i = 0; i < N; i++) {
        const float *inp_row = inp + i * C; //从 inp 的起始位置，向后跳过 i 行，每行有 C 个 float，所以总共跳 i*C 个 float 元素 ，这个位置就是 第 i 行的第 0 列的地址。
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {   //先遍历找最大值
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval); //计算 e^{z_{i}-max}
            sum += out_row[j];
        }
        float norm = 1.f / sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

// CUDA kernel
__global__ void softmax_forward_kernel1(float *out, const float *inp, int N,
                                        int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float) sum;
        }
    }
}

// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) {
    for (int i = 0; i < N * C; ++i) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                      << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                      << std::endl;
            return false;
        }
    }
    return true;
}

__global__ void softmax_forward_kernel2(float *out, const float *inp, int N,
                                        int C) {
    extern __shared__ float shared[];//extern是声明 动态的共享内存
    int idx = blockIdx.x;   // ranges [0, N)
    int tid = threadIdx.x;  // ranges [0, block_size)
    int block_size = blockDim.x;
    const float *x = inp + idx * C;  // idx-th row of inp
    // thread coarsening
    float maxval = -INFINITY;
    //假设每行512个元素，每个线程块16个，则每个线程负责的数据为 512/16 =32
    //例如，线程 0 处理索引为 0、16、32、64、… 的元素，线程 1 处理索引为 1、17、33、65、… 的元素，以此类推。通过这种方式，每个线程在其所负责的数据范围内独立计算出一个局部最大值和一个局部和。最终，我们共获得 16 个局部最大值和 16 个局部和。
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval; //每个线程都这样，也就是说会有16个线程
    //线程的执行是异步的：
    //有些线程可能已经写完 shared[tid]
    //有些线程可能还没写完
    //如果直接往下跑，有线程会读到还没写好的数据，就会出现错误。
    __syncthreads();
    // reductions 已知shared[tid] 里存放了每个线程的局部最大值， 一个block里有block_size个线程，所以 shared 里有 block_size 个值；
    for (int stride = block_size / 2; stride >= 1; stride /= 2) { //从一半开始，每次除以 2，直到 1。只有一半线程参与计算
        __syncthreads();
        if (tid < stride) { //控制哪些线程参与当前轮的比较 避免重复计算或访问越界。
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0]; //拿到最大值了
    // compute expf and write the result to global memory
    //线程0处理索引为 0、16、32、64、… 的元素，线程 1 处理索引为 1、17、33、65、… 的元素，以此类推。
    for (int i = tid; i < C; i += block_size) { //tid 当前线程在block的id
        out[idx * C + i] = expf(x[i] - offset); //idx代表第几行
    }
    __syncthreads();
    // thread coarsening again, for the sum
    x = out + idx * C;  // idx-th row of out  开始指向 inp 里第 idx 行的第一个元素
    float sumval = 0.0f; //每个线程 计算它负责的那部分的和
    for (int i = tid; i < C; i += block_size) {//线程0处理索引为 0、16、32、64、… 的元素，线程 1 处理索引为 1、17、33、65、… 的元素，以此类推。
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}

//__device__定义 GPU 内部函数，只能被其他 GPU 函数（__device__ 或 __global__）调用，不能直接从 CPU 调用
//__global__定义核函数（kernel），是 CUDA 启动的入口函数，必须用 <<< >>> 语法启动，返回类型必须是 void
//第 1 轮（offset = 16） 每个线程会和它 +16 的线程比：
//lane 0 和 lane 16 比，取大值
//lane 1 和 lane 17 比
//lane 2 和 lane 18 比
//lane 15 和 lane 31 比
//执行后
//0: max(val[0], val[16])
//1: max(val[1], val[17])
//15: max(val[15], val[31])
//16~31: 值可能没用到，但它们也会被赋新值
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

//把一个warp内的所有线程的值加在一起
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel3(float *out, const float *inp, int N,
                                        int C) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;

    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    //这里要不要同步？
    //要时刻理解 这不是循环 而是每个tid执行一个
    maxval = warpReduceMax(maxval); //每个线程都会调一遍 每次都会操作所有线程 不重复吗？
    //warpReduceMax 内部的 __shfl_down_sync 是硬件 warp shuffle 指令，它直接在 warp 内线程寄存器之间交换数据，并行发生，不是循环挨个去操作别人的值。
    //所以不是“一个线程处理所有线程”，而是“所有线程在同一条指令下交换和比较各自的值”。
    //为什么不重复？假设 warp 有 32 个线程，offset = 16：
    //lane0 比较自己和 lane16 的值
    //lane1 比较自己和 lane17 的值
    //lane2 比较自己和 lane18 的值
    //lane15 比较自己和 lane31 的值
    //lane16~31 拿到的 partner 超出范围，不更新
    //这样一次循环就完成了半个 warp 的比较，完全并行，没有重复处理同一对值。

    //
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);//而是在把最大值广播给 warp 中所有线程。上边操作没改变值？从 lane 0 把最大值广播给 warp 全员

    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) { //求每个线程负责的 最大值
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);

    float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);

    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

__global__ void softmax_forward_kernel4(float *out, const float *inp, int N,
                                        int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for
    // intra-warp reductions shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;  // warp index within a block
    int laneId = threadIdx.x % 32;  // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float *maxvals = shared;
    float *sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float *x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

int main() {
    // Example: batch size N=32, classes C=4096
    int N = 320;
    int C = 4096;

    size_t num_elements = N * C;
    float *inp = (float *) malloc(num_elements * sizeof(float));
    float *out_cpu = (float *) malloc(num_elements * sizeof(float));
    float *out_gpu = (float *) malloc(num_elements * sizeof(float));

    // Initialize input with sample data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            inp[n * C + c] = float(c);
        }
    }

    // Run CPU version and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, inp, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Run GPU version and measure time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_out, *d_inp;
    cudaMalloc((void **) &d_out, N * C * sizeof(float));
    cudaMalloc((void **) &d_inp, N * C * sizeof(float));
    cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // Launch kernel
    int blockSize = 128;
    int numBlocks = N;
    softmax_forward_kernel3<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate milliseconds
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copy result back to host
    cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_inp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compare results
    bool success = compare_results(out_cpu, out_gpu, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

    // Print performance comparison
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
              << std::endl;

    // Cleanup
    free(inp);
    free(out_cpu);
    free(out_gpu);

    return 0;
}
