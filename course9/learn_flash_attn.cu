#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include "helper.h"


using FP=float;
const int Br =2;
const int Bc =2;

const int input_seq =4;

const int dim =4;

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
__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O,int seqlen,float smScale) {
    // kv  的块规模
    //group of row(seqlen)
    int groupSeq = (seqlen + Bc - 1) / Bc;  //这里应该是 对 kv块 要处理的批次

    int groupTx = (dim+Bc-1)/Bc;
    int groupTy = (dim+Br-1)/Br;  //就这两变量不太懂反正

    __shared__ float sQ[Br][dim];
    __shared__ float sK[Bc][dim];
    __shared__ float sV[Bc][dim];

    __shared__ float sO[Br][dim];
    __shared__ float sQK[Br][Bc];

    __shared__ float sSafeE[Br][Bc];

    __shared__ float sDenom[Br];
    __shared__ float sMax[Br];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y*blockDim.y+ ty;
    if(row >=seqlen){
        return;  //这里很容易理解 就是如果当前线程的行号大于序列长度就返回
    }
    //加载Q O 到共享内存

}


void flash_attention_v2_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
    float sm_scale = 1.f /sqrtf(static_cast<float>(n));
    int BS = 1;
    int HEAD = 1;
    int SEQLEN = m;
    int DIM = n;

    //网格维度
    int Gc = 1;
    int Gr = (SEQLEN+ Br -1)/Br;

    dim3 grid = dim3(Gc, Gr);

    dim3 block = dim3(Br, Bc);

    flash_attention_v2_kernel<<<grid,block>>>(Q,K,V,O,SEQLEN,sm_scale);

    DEBUG_BLOCK(printf("==v2:0 ==\n"); print_device_matrix(O, m, n););


  }



void test_attention(){
    int m = input_seq;
    int n = dim;

    float* h_K = new float[m*n];
    float* h_V = new float[m*n];
    float* h_Q = new float[m*n];
    float* h_O = new float[m*n];
    float* h_O2 = new float[m*n];

    //初始化 Ｑ　Ｋ　Ｖ
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

    float *d_K ,*d_Q,*d_V, *d_O, *d_O2;
    cudaMalloc((void **) &d_K, sizeof(float) * m * n);
    cudaMalloc((void **) &d_Q, sizeof(float) * m * n);
    cudaMalloc((void **) &d_V, sizeof(float) * m * n);
    cudaMalloc((void **) &d_O, sizeof(float) * m * n);
    cudaMalloc((void **) &d_O2, sizeof(float) * m * n);

    //将数据拷贝至GPU
    cudaMemcpy(d_K, h_K, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //执行测试
    for(int i=0;i<1;i++){
        self_attention_cuda(d_Q,d_K,d_V,d_O,m,n);
        CUDA_CHECK(cudaGetLastError());
    }


    // TEST FLASH ATTENTION2
    for (int i = 0; i < 1; i++) {
        flash_attention_v2_cuda(d_Q, d_K, d_V, d_O2, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Time for kernel execution: %.3f ms \n",milliseconds/100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //将结果拷贝回主机
    cudaMemcpy(h_O, d_O, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O2, d_O2, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    printf("== O ==\n");
    bool res = all_close(h_O, h_O2, m, n);
    if(res){
        printf("is equal\n");
    }
    else{
        printf("is not equal\n");
    }
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_O2);
    free(h_K);
    free(h_V);
    free(h_Q);
    free(h_O);
    free(h_O2);
}





int main()
{
    int epoch =10;
    for(int i=0;i<epoch;i++){
        test_attention();
    }
    return 0;
}