#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include "helper.h"

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf("CUDA_CHECK error in line %d of file %s: %s\n", __LINE__, \
             __FILE__, cudaGetErrorString(cudaGetLastError()));        \
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
bool read_bin(const char*filename, float *h_data,size_t num_elements){
    std::ifstream file(filename,std::ios::binary);
    if(!file){
        printf(" failed to open file %s\n",filename);
        return false;
    }
    file.read((char*)h_data,num_elements*sizeof(float));
    if(!file){
        printf(" failed to read file %s\n",filename);
        file.close();
        return false;
    }
    file.close();
    printf("loaded %s  (%zu elements)",filename,num_elements);
    return true;
}

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M , int N,int K ,int mBlock){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= mBlock; //这里乘mBlock的作用是将线程号映射成起始行号。比如0号线程负责从0行开始，1号线程，负责从mBlock行开始，2号线程从2*mBlock开始
    for (int i = idx; i < idx + mBlock; i++) {
        //不对啊 这个idx+ mBlock对吗？  这里是+2 啊
        //这里很简单 单纯的矩阵相乘  对于A中的一行， 遍历B中的每一列 （遍历每一列中的每一个元素）
        for(int j=0 ; j<N; j++){
            float sum = 0.f;
            for(int k=0; k<K; k++){
                //你还别说 这里的B可能没有转置
                sum += A[i*K+k]*B[j*K+k];
            }
            C[i*N+j] = a*sum + b*C[i*N+j];
        }
    }
}

__global__ void row_softmax(float *input, float *output, int n) {
    //softmax = (e^{x_i}-e^{x_max})/(sum(  e^{x_i}-e^{x_max}   ))
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float max_val = -INFINITY;
    float sum = 0.f;
    //这里就是找到每一行的最大值
    for (int i = 0; i < n; i++) {
        if (input[idx * n + i] > max_val) {
            max_val = input[idx * n + i];
        }
    }
    //上边是找到最大值了
    for(int i = 0; i < n; i++) {
        //此处求 e^{x_i}-e^{x_max}
        output[idx*n+i] = expf(input[idx * n + i] - max_val);
        sum += output[idx*n+i];
    }
    //每个位置再除以总和
    for(int i =0 ; i<n ;i++){
        output[idx*n+i] /= sum;
    }

}

__global__ void naive_pv(float *P, float *V, float *O, int M, int N,int mBlock) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    idx *= mBlock; //同样的 每个idx负责mBlock行 这样起始地址就变为了idx
    int K = M;
    //这个写法跟 naive_nrow_gemm 一样
    for(int i = idx; i < idx + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += P[i * K + k] * V[k * N + j];

            }
            O[i * N + j] = sum;
        }
    }



}


bool write_bin(const char* filename, const float *h_data, size_t num_elements){
    std::ofstream file(filename,std::ios::binary);
    if(!file){
        printf("failed to create file %s\n",filename);
        return false;
    }
    file.write((const char*)h_data,num_elements*sizeof(float));
    file.close();
    printf("saved %s (%zu elements)\n",filename,num_elements);
    return true;
}

void self_attention_cuda(float*Q,float*K ,float *V,float *O, int m, int n){
    int mBlock =2;
    assert(m % mBlock == 0 && "mBlock should align") ;
    float sm_scale = 1.f / sqrtf(static_cast<float>(n));
    float *sm_o ;
    cudaMalloc((void **)&sm_o, m * m * sizeof(float));


    dim3 qk_block(m / mBlock, 1,1);
    naive_nrow_gemm<<<1,qk_block>>>(Q, K, sm_o, sm_scale, 0,m,m, n, mBlock); //这里应该是对Q和K进行矩阵乘法，得到QK矩阵
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");print_device_matrix(sm_o, m, m););

    //获得了QK[m,m]
    dim3 sm_block(m, 1,1); //这里就是 m个线程负责softmax归一化
    row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("==naive softmax QK ==\n");print_device_matrix(sm_o, m, m););


    //计算 QK[m,m] V[m,n]
    dim3 qkv_block(m / mBlock, 1,1);
    naive_pv<<<1,qkv_block>>>(sm_o,V,O,m,n ,mBlock);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError();printf("===== naive softmax(QK)V ==\n");print_device_matrix(O, m, n););



    cudaFree(sm_o);
}

void self_attention_with_io(int m, int n){
    size_t num_elements = m*n;

    //CPU内存分配
    float *h_Q = new float[num_elements];
    float *h_K = new float[num_elements];
    float *h_V = new float[num_elements];
    float *h_O = new float[num_elements];

    read_bin("/cuda_code/course9/tmp/Q.bin",h_Q, num_elements);
    read_bin("/cuda_code/course9/tmp/K.bin",h_K, num_elements);
    read_bin("/cuda_code/course9/tmp/V.bin",h_V, num_elements);

    //GPU内存分配
    float *d_Q, *d_K ,*d_V, *d_O;
    cudaMalloc(&d_Q, num_elements*sizeof(float));
    cudaMalloc(&d_K, num_elements*sizeof(float));
    cudaMalloc(&d_V, num_elements*sizeof(float));
    cudaMalloc(&d_O, num_elements*sizeof(float));

    //把数据挪到GPU上
    cudaMemcpy(d_Q, h_Q, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, num_elements*sizeof(float), cudaMemcpyHostToDevice);

    // run self-attention
    self_attention_cuda(d_Q,d_K, d_V, d_O, m, n);

    //把结果拷贝回CPU
    cudaMemcpy(h_O, d_O, num_elements*sizeof(float), cudaMemcpyDeviceToHost);

    write_bin("/cuda_code/course9/tmp/O_cuda.bin", h_O, num_elements);

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    printf("🎉 Self-attention completed. Output saved to O_cuda.bin\n");

}













int main(){
    const int m = 64;
    const int n =128;

    printf("Running self-attention for m=%d, n=%d\n", m, n);
    self_attention_with_io(m,n);

    return 0;
}