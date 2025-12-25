#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>

#include "helper.h"

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf(                                                          \
          "CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                \
          __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// #define DEBUG
// 如果定义了，则DEBUG_BLOCK(expr) 会真正执行 expr 里的代码。
#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
  do {                    \
    expr                  \
  } while (0)
#else
#define DEBUG_BLOCK(...) \
  do {                   \
  } while (0)
#endif

// data type to test
using FP = float;
// BLOCK_M(Br, Brow), BLOCK_N(Bc, Bcol) can be determined at compile time
// just like offical implementation which use a template kernel to do that
// Block row size
const int Br = 2;
// Block column size
const int Bc = 2;
// seqlen
const int input_seq = 4;
// dim
const int dim = 4;

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock);

__global__ void row_softmax(float *input, float *output, int n);

__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock);

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O,
                                          int seqlen, FP smScale);

void flash_attention_v2_cuda(FP *Q, FP *K, FP *V, FP *O, int m, int n) {
    FP sm_scale = 1.f / sqrtf(static_cast<FP>(n)); //static_cast<FP>(n)将整数 n转换为 FP类型（可能是浮点数）
    int BS = 1;
    int HEAD = 1;
    int SEQLEN = m;
    int DIM = n;

    //计算网格维度
    int Gc = 1; //列方向的网格块数为1
    int Gr = (SEQLEN + Br - 1) / Br; //行方向的网格块数  为了确保覆盖整个序列长度

    // NOTE: each block process a range row of Q
    dim3 grid = dim3(Gc, Gr);
    // NOTE: each thread process a tile of Q  每
    dim3 block = dim3(Bc, Br); //Br 的r代表  row  , Bc 的c代表column
    flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, SEQLEN, sm_scale);

    DEBUG_BLOCK(printf("== v2: O ==\n"); print_device_matrix(O, SEQLEN, DIM););
}

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O,
                                          int seqlen, FP smScale) {
    // block size for K, V
    // group of row(seqlen)
    int groupSeq = (seqlen + Bc - 1) / Bc;
    // parallel process for V[Br, d]
    // group of column
    //线程块大小 设定为  Br * Bc
    int groupTx = (dim + Bc - 1) / Bc; //这里用 dim 除以Bc 代表
    int groupTy = (dim + Br - 1) / Br;

    //声明共享显存数组 用于高效地从全局显存中加载数据片段。  这是块内共享的
    // 每个 SM 最多 32 个 block。__shared__ 表示 共享内存（Shared Memory），存在于 SM（Streaming Multiprocessor）片上 SRAM。
    // load slice from global memory(HBM)
    __shared__ FP sQ[Br][dim]; //这些 __shared__ 数组只对 同一个 block 内的线程可见。
    __shared__ FP sK[Bc][dim];
    __shared__ FP sV[Bc][dim];
    // tmp o
    __shared__ FP sO[Br][dim];
    __shared__ FP sQK[Br][Bc]; //用来存储当前分块计算得到的注意力打分矩阵S = Q*K^T,softmax之前的
    // e^{x - max}
    __shared__ FP sSafeE[Br][Bc];//e^{x - local max}
    // s stand for shared and local
    __shared__ FP sDenom[Br];//全局归一化分母缓存
    __shared__ FP sMax[Br]; //存储max(所有已见的 QK^T)

    // TODO: multihead

    // [0, Bc]
    int tx = threadIdx.x; //tx 负责大教室的某一行 Q O的行
    // [0, Br]
    int ty = threadIdx.y; //负责K V的某一行

    //当前线程负责的全局行号 = block 内的行号 + block 在网格里的偏移。
    //blockIdx.y：当前 block 在 y 方向的索引。blockDim.y：每个 block 在 y 方向上有多少线程（就是 Br）。ty：当前线程在 block 内的行坐标。
    int row = ty + blockIdx.y * blockDim.y; //相当于二维转成一维了，相当于当前学生在 块内的全局行号
    if (row >= seqlen) {
        return;
    }
    // load q, o, max, denom from global memory to shared memory
    for (int i = 0; i < groupTx; i++) {
        //sQ[ty][:]：共享内存里存的是 Q 的一行。ty：表示是哪一行（由 threadIdx.y 决定）
        //i * Bc + tx：表示这一行的第几列。
        //共享内存 sQ 第 ty 行、第 (i*Bc+tx) 列
        //← 拷贝自全局内存 Q 的第 row 行、第 (i*Bc+tx) 列。
        sQ[ty][i * Bc + tx] = Q[row * dim + i * Bc + tx];
        // NOTE:: accumulator zero init here
        sO[ty][i * Bc + tx] = 0;
    }

    sMax[ty] = -INFINITY;
    sDenom[ty] = 0;

    // load K, V block
    // Q[Br][dim] @ K[0..seqlen.step(Bc), dim]
    // compute partial sum of O[ty][dim] each iteration
    for (int j = 0; j < groupSeq; j++) {
        if ((j * Bc + tx) < seqlen) {
            // load k, v from global memory to shared memory
            // K[seqlen, dim], V[seqlen, dim]
            for (int i = 0; i < groupTy; i++) {
                sK[tx][i * Br + ty] = K[j * Bc * dim + tx * dim + i * Br + ty];
                sV[tx][i * Br + ty] = V[j * Bc * dim + tx * dim + i * Br + ty];
            }
        }

        // wait until g2s done
        __syncthreads();

        // compute qk
        FP sum = 0.f;
        // result oriented: qk[y][x] from q[y] @ k[x]
        for (int i = 0; i < dim; i++) {
            sum += sQ[ty][i] * sK[tx][i];
        }
        // sQK[Br, Bc]
        sQK[ty][tx] = sum * smScale;

        // wait until qk done
        __syncthreads();

        // compute local max of each row of qk
        FP localMax = -INFINITY;
        for (int i = 0; i < Bc; i++) {
            localMax = max(localMax, sQK[ty][i]);
        }
        __syncthreads();
        // compute the max of each row
        FP newMax = max(sMax[ty], localMax);

        // compute safe e(e^{x - max}) of each qk element
        sSafeE[ty][tx] = exp(sQK[ty][tx] - newMax); //计算 e^(x-max) → 存入 sSafeE
        __syncthreads();

        // accumulate local denom of each row of qk with local max
        FP localDenom = 0.f;
        for (int i = 0; i < Bc; i++) {
            localDenom += sSafeE[ty][i];
        }
        __syncthreads();

        // rescale history result
        FP rescaleOld = exp(sMax[ty] - newMax);
        // rescale denom
        FP newDenom = sDenom[ty] * rescaleOld + localDenom;

        // NOTE:
        // QK[Br, Bc] @ V[Bc, d] = O[Br, d]
        // tx in [0, Bc], ty in [0, Br]
        // slice-Bc and each O[ty, group.x] as accumulator
        for (int i = 0; i < groupTx; i++) {
            // NOTE: rescale old_o(numerator only for now) once: old_nume * rescale
            sO[ty][i * Bc + tx] = (sO[ty][i * Bc + tx] * rescaleOld);
            for (int k = 0; k < Bc; k++) {
                // NOTE:
                // accumulate numerator
                // new_nume = old_nume' + local_nume (Softmax(QK)@V)
                sO[ty][i * Bc + tx] += sSafeE[ty][k] * sV[k][i * Bc + tx];
            }
        }

        // update global max and denom
        sMax[ty] = newMax;
        sDenom[ty] = newDenom;
        __syncthreads();
    }

    // rescale O in the end
    for (int i = 0; i < groupTx; i++) {
        // copy sO[row, dim] to gO[row, dim]
        O[row * dim + i * Bc + tx] = sO[ty][i * Bc + tx] / sDenom[ty];
    }
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
    int mBlock = 2;
    assert(m % mBlock == 0 && "mBlock should align");

    float sm_scale = 1.f / sqrtf(static_cast<float>(n));
    float *sm_o;
    cudaMalloc((void **) &sm_o, sizeof(float) * m * m);

    dim3 qk_block(m / mBlock, 1, 1);
    naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");
                        print_device_matrix(sm_o, m, m););

    // QK[M, M]
    dim3 sm_block(m, 1, 1);
    row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
                        printf("== naive softmax(QK) ==\n");
                        print_device_matrix(sm_o, m, m););

    // QK[M, M] @ V[M, N]
    dim3 qkv_block(m / mBlock, 1, 1);
    naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
                        printf("== naive softmax(QK)V ==\n");
                        print_device_matrix(O, m, n););

    cudaFree(sm_o);
}

// naive gemm implement with slice-k
// perform C = aA@B + bC
// A[M, K] x B[K, N] = C[M, N]
// each thread process mblock rows of A
__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // each thread process a range of rows
    idx *= mBlock;

    // A[mBlock, K] x B[N, K].T = C[mBlock, N]
    for (int i = idx; i < idx + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            // C[M, N]
            // C = aA@B + bC
            C[i * N + j] = a * sum + b * C[i * N + j];
        }
    }
}

// perform QK[M, M] @ V[M, N]
__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // each thread process a range of rows
    idx *= mBlock;

    int K = M;
    // P[mBlock, M] x V[M, N] = O[mBlock, N]
    for (int i = idx; i < idx + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += P[i * K + k] * V[k * N + j];
            }
            // C[M, N]
            O[i * N + j] = sum;
        }
    }
}

// each thread process one row of softmax
__global__ void row_softmax(float *input, float *output, int n) {
    // assume id will not exceed row number of input
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float max = -INFINITY;
    float sum = 0.f;

    // Find max
    for (int i = 0; i < n; i++) {
        if (input[idx * n + i] > max) {
            max = input[idx * n + i];
        }
    }

    // Compute numerator and denominator
    for (int i = 0; i < n; i++) {
        output[idx * n + i] = exp(input[idx * n + i] - max);
        sum += output[idx * n + i];
    }

    // Compute softmax
    for (int i = 0; i < n; i++) {
        output[idx * n + i] /= sum;
    }
}

void test_attention() {
    // seqlen
    int m = input_seq;
    // dim
    int n = dim;

    // Host pointer
    float *h_K = new float[m * n];
    float *h_Q = new float[m * n];
    float *h_V = new float[m * n];
    float *h_O = new float[m * n];
    float *h_O2 = new float[m * n];

    // 初始化 K, Q, V
    // 这里是随机初始化，rand()返回一个随机整数
    for (int i = 0; i < m * n; ++i) {
        //除以 RAND_MAX 把它缩放到 [0,1] 区间。
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
        //如果开启 DEBUG_BLOCK 宏（调试模式），就不会用随机数，而是直接用序号 i 初始化。
        DEBUG_BLOCK(h_K[i] = static_cast<float>(i); h_Q[i] = static_cast<float>(i);
                            h_V[i] = static_cast<float>(i););
    }

    DEBUG_BLOCK(printf("== K ==\n"); print_host_matrix(h_K, m, n););

    float *d_K, *d_Q, *d_V, *d_O, *d_O2;
    // Malloc device memory
    //这里为什么做了一个指针类型的转换？
    cudaMalloc((void **) &d_K, sizeof(float) * m * n);
    cudaMalloc((void **) &d_Q, sizeof(float) * m * n);
    cudaMalloc((void **) &d_V, sizeof(float) * m * n);
    cudaMalloc((void **) &d_O, sizeof(float) * m * n);
    cudaMalloc((void **) &d_O2, sizeof(float) * m * n);

    // Copy data from host to device
    cudaMemcpy(d_K, h_K, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Run test
    for (int i = 0; i < 1; i++) {
        // Launch kernel
        self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

        CUDA_CHECK(cudaGetLastError());
    }

    // test flash attention 2
    for (int i = 0; i < 1; i++) {
        flash_attention_v2_cuda(d_Q, d_K, d_V, d_O2, m, n);
        //cudaGetLastError()  CUDA Runtime API 的函数  返回最近一次 CUDA 调用（尤其是 内核调用 kernel<<<>>>）的错误状态。
        //如果之前没有错误，它会返回 cudaSuccess。如果有错误，比如非法内存访问、线程配置错误等，它会返回一个错误码（cudaError_t 类型）。
        CUDA_CHECK(cudaGetLastError());
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Time for kernel execution: %.3f ms \n", milliseconds / 100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Result back to host
    cudaMemcpy(h_O, d_O, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O2, d_O2, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    bool res = all_close(h_O, h_O2, m, n);
    if (res) {
        printf("is equal\n");
    } else {
        printf("is not equal\n");
    }
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_O2);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O2);
}

int main() {
    int epoch = 10;
    for (int i = 0; i < epoch; i++) {
        test_attention();
    }

    return 0;
}