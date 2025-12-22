#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

//__global__ void test_shuf_broadcast(int *dOutput, int *dInput, int srcLane) {
//    int val = dInput[threadIdx.x];
//    val = __shfl_sync(0xffffffff, val, srcLane,32);
//    dOutput[threadIdx.x] = val;
//}
//
//
//
//int main(){
//    const int numThreads = 32;
//    const int srcLane = 2;
//    int hInput[numThreads];
//    int hOutput[numThreads];
//    for (int i = 0; i < numThreads; ++i) {
//        hInput[i] = i;
//    }
//    int *dInput, *dOutput;
//    cudaMalloc(&dInput, numThreads * sizeof(int));
//    cudaMalloc(&dOutput, numThreads * sizeof(int));
//
//    cudaMemcpy(dInput, hInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);
//
//    test_shuf_broadcast<<<1, numThreads>>>(dOutput, dInput, srcLane);
//
//    cudaMemcpy(hOutput, dOutput, numThreads * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < numThreads; ++i) {
//        std::cout << hOutput[i] << " ";
//    }
//    cudaFree(dInput);
//    cudaFree(dOutput);
//    return 0;
//
//}
//
void softmax_forward_cpu(float *out, const float *input, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float *inp_row = input + i * C;
        float *out_row = out + i * C;
        float maxval = -INFINITY;
        float sum = 0.f;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        for (int j = 0; j < C; j++) {
            out[j] = expf(inp_row[j] - maxval);
            sum += out[j];
        }
        float norm = 1.f / sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }

    }
}
//该算子每行配一个独立的线程块 即共配置N个线程块
//每个线程块中仅包含一个线程。由这个唯一的线程负责完成对应行所有数据的归约计算（包括求最大值和指数和）
//看自己的笔记 这个函数就相当于去掉外层循环呗
__global__ void softmax_forward_kernel1(float *out, const float *input, int N, int C) {
    //
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        const float *inp_row = input + i * C;
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
            out_row[j] /= sum;
        }

    }
}

__global__ void softmax_forward_kernel2(float *out, const float *input, int N, int C) {
    //这个整体上就是  N个线程块 ，每个线程负责16个线程，这一个线程块负责 C个数据
    extern __shared__ float shared[];
    //每个线程负责32个数据，已知C为512 ，那么需要有16个线程负责这一行
    //先不准用各种同步原语实现
    //要求每个线程负责的最大值，要求每个线程负责的值 的指数之和
    int block_size = blockDim.x; //这里其实是16
    int idx = blockIdx.x;
    int tid = threadIdx.x;

    float maxval = -INFINITY;
    const float *x = input + idx * C;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }//找到每个线程负责的局部最大值了
    shared[tid] = maxval;//存到共享显存里边去
    __syncthreads();//同步

    //该线程块包含16个局部最大值。
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }//找到每个线程块的最大值了
    __syncthreads();
    //算每个数据的指数
    float offset = shared[0];
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();

    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval; //求每个线程负责的局部和

    //规约求总和
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();
    float norm = 1.f / shared[0];
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] *= norm;

    }
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

//这个是一个块负责一个
__global__ void softmax_forward_kernel3(float *out, const float *input, int N, int C) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = input + idx * C; //这里这不还是直接读取显存的

    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]); //每个线程拿到自己负责的一个局部最大值，现在warp里都塞满了局部最大值
    }
    maxval = warpReduceMax(maxval); //通过自定义函数获取全局最大值；这条指令是不是 每个线程都执行？这样不重复吗？

    //不明白  这里为啥广播？ 不广播不行吗？ 反正上边都拿到了全局最大值
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0); //把最大值广播给warp内所有线程

    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset); //就是我上边不广播  直接用哪个maxval不行？
    }

    //现在要求和了  跟求最大值一样的套路  先每个线程拿到局部和，然后再规约求和
    float sumval = 0.0f;
    x = out + idx * C;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += out[i];
    }
    //此时求得全局和，同样的疑问 这岂不是每个线程都执下边这一句了？不重复吗？
    sumval = warpReduceSum(sumval);

    float sum = __shfl_sync(0xFFFFFFFF, sumval, 0); //广播全局最大值

    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] /= sum;
        //我这里直接  out[idx * C + i] /= sumval;不行吗？ 不行的话  那这里为啥要广播？

    }

}

__global__ void softmax_forward_kernel4(float *out, const float *input, int N, int C) {
    //每个线程块包含4个 warp ，128个线程， 可用共享内存 ，写吧
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warpsPerBlock = blockDim.x / 32; //每个线程块包含4个warp

    float *maxvals = shared; //该行等价于float *maxvals = &shared[0];  即 让指针 maxvals 指向共享内存的起始位置”。
    float *sumvals = &shared[warpsPerBlock]; //这两句什么意思 怎么一个带取址符 一个不带？
//    意思是：
//    “让指针 sumvals 指向共享内存中第 4 个 float 的位置”。
//    即内存布局如下：
//    shared:
//    索引:   [0] [1] [2] [3] [4] [5] [6] [7]
//           |--maxvals-----|--- sumvals ---|
    const float *x = input + idx * C;

    //先求最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);//求每个线程负责的最大值，
    }
    maxval = warpReduceMax(maxval);//求warp内部的最大值

    if (laneId == 0) maxvals[warpId] = maxval;//每个warp的0号线程负责把最大值写入共享内存

    __syncthreads();//尤其不理解这个同步的意义  ， kernel3中就没有这个同步吧

    //求全局最大值 , 共享显存里存着4个warp的最大值，比较这几个 取最大的
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();//尤其不理解为什么同步

    float offset = maxvals[0];
    //计算指数
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    //求和
    float sumval = 0.0f;
    x = out + idx * C;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval); //线程0的拿到最大值
    if (laneId == 0) sumvals[warpId] = sumval; //这句如果不写if 都执行会怎样？ 看上句子注释 必须if
    __syncthreads();//这里不知道为啥同步 是因为上边写if的原因吗？ kernel3里也没同步啊

    //这里求全局和 ，但是我没有if tid==0啊
//    for(int i=0; i<warpsPerBlock; i++) {
//        sumval += sumvals[i];
//        }
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 0; i < warpsPerBlock; i++) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads(); //不理解为什么同步
    float norm = 1.f / sumvals[0];
    for (int i = 0; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] * norm;
    }


}

__global__ void softmax_forward_kernel4_copy(float *out, const float *input, int N, int C) {
    //每个线程块包含4个 warp ，128个线程， 可用共享内存 ，写吧
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32; //每个线程块包含4个warp

    float *maxvals = &shared[0];
    float *sumvals = &shared[warpsPerBlock];

    //先求最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, input[idx * C + i]);//每个线程留存自己的最大值
    }
    maxval = warpReduceMax(maxval);//每个warp的0号线程负责把最大值写入共享内存
    if(laneId==0){
        maxvals[warpId] = maxval;
    }
    __syncthreads();

    //求全局最大值
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]); //这里不还是直接访问的全局显存？这样不还是慢？不能先弄到共享内存里来？
        }
        maxvals[0] = val; //将全局最大值写道共享内存里去
    }
    float offset = maxvals[0];

    //求和 先每个线程求和

    for (int i = tid; i < C; i += blockDim.x) {
        out[idx*C+i] = expf(input[idx*C+i] - offset);
    }
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += out[idx*C+i];
    }
    sumval = warpReduceSum(sumval); //线程0的拿到最大值
    if(laneId==0){
        sumvals[warpId] = sumval;
    }
    __syncthreads();
    if(tid==0){
    for(int i=0; i<warpsPerBlock; i++) {
        sumval += sumvals[i];
    }
    sumvals[0]= sumval;
    }
    __syncthreads();
    float norm = 1.f / sumvals[0];
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] *= norm;
    }

}
int main() {
    int N = 320;
    int C = 2048;

    size_t num_elements = N * C;
    size_t bytes = num_elements * sizeof(float);

    float *inp = (float *) malloc(bytes);
    float *out_cpu = (float *) malloc(bytes);
    float *out_gpu = (float *) malloc(bytes);//用于放GPU的执行结果

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            inp[n * C + c] = float(c);
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, inp, N, C);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end_time - start_time;
    std::cout << "CPU time : " << duration.count() << std::endl;





    //GPU版本
    float *d_inp, *d_out;
    cudaMalloc((void **) &d_inp, bytes);
    cudaMalloc((void **) &d_out, bytes);
    cudaMemcpy(d_inp, inp, bytes, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop, start2, stop2, start3, stop3,start4,stop4;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);




    //1
    int blockSize = N;
    int numBlocks = 1;
    cudaEventRecord(start);
    softmax_forward_kernel1<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0.f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    //2
    cudaEventRecord(start2);
    int blockSize_2 = 128; //每个线程处理32个数据
    int numBlocks_2 = N;
    softmax_forward_kernel2<<<numBlocks_2, blockSize_2>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float gpu_time_ms2 = 0.f;

    cudaEventElapsedTime(&gpu_time_ms2, start2, stop2);
    //3
    cudaEventRecord(start3);
    int blockSize_3 = 32; //每个线程处理32个数据
    int numBlocks_3 = N;
    softmax_forward_kernel3<<<numBlocks_3, blockSize_3>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float gpu_time_ms3 = 0.f;
    cudaEventElapsedTime(&gpu_time_ms3, start3, stop3);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaEventRecord(start4);
    int blockSize_4 = 128;
    int numBlocks_4 = N;
    softmax_forward_kernel4<<<numBlocks_4, blockSize_4>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);
    float gpu_time_ms4 = 0.f;
    cudaEventElapsedTime(&gpu_time_ms4, start4, stop4);


    std::cout << "GPU1 time : " << gpu_time_ms << std::endl;
    std::cout << "GPU2 time : " << gpu_time_ms2 << std::endl;
    std::cout << "GPU3 time : " << gpu_time_ms3 << std::endl;
    std::cout << "GPU4 time : " << gpu_time_ms4 << std::endl;
    cudaMemcpy(out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    cudaEventDestroy(start4);
    cudaEventDestroy(stop4);
    free(inp);
    free(out_cpu);
    free(out_gpu);
    return 0;


}
