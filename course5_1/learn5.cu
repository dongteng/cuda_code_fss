#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>

#define BLOCK_SIZE 32
#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B,float beta, float* C) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y* blockDim.y + threadIdx.y;
    //C[gy][gx]
    //写这个矩阵相乘 首先需要明确的是 已经有了gx gy了，那么该线程就负责 A中第 gy行，与B中第gx列的


    if (gx >= N || gy >= M) return;
    float tmp = 0.0f;       //累加器，保存 A 的一行 与 B 的一列 点积结果
    for (int i = 0; i < K; i++) {
        tmp += A[gy*K+i]*B[gx+i*N];
    }
    C[gy*N+gx]= alpha*tmp + beta*C[gy*N+gx];

}
//__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,float beta, float* C) {
//    //写这个的总体思路便是不如直接类比 之前线程 (gx,gy)负责 C[gy][gx]
//    //这里同样，线程块（blockIdx.x, blockIdx.y）负责 C[blockIdx.y, blockIdx.x] 这个 C 分块的大小就是 [BM,BN]
//    //要使用共享显存
//    int bx = blockIdx.x;
//    int by = blockIdx.y;
//
//    const int BM = BLOCK_SIZE;
//    const int BN = BLOCK_SIZE;
//    const int BK = BLOCK_SIZE;
//
//    int tx = threadIdx.x % BN;
//    int ty = threadIdx.y / BN; //因为这个线程块是一维的，所以只有threadIdx.x没有 y，但是矩阵快运算是2维的，所以需要映射
//
//    __shared__ float As[BM*BK];
//    __shared__ float Bs[BN*BK];//注意此处 动态只能设置一个，静态可以有多个！
//
//    //这里需要明确的是，线程块负责的是 C[by][bx] 这个分块，每个分块大小是BM * BN
//
//    // 🔹 A 的当前子块起始位置：
//    A = &A[by*BM*K] ;//此处看ppt画的图吧
//    B = &B[bx*BN];
//    C = &C[by*BM*N+bx*BN];
//
//    // ❗每个线程要计算 C 子块中一个具体的元素 Csub[ty, tx]
//    // 因此每个线程最终要累加 BK 次乘法结果。
//    float tmp = 0.f;
//    //这单个线程还要干什么呢？ 一个线程块求的是C[bx][by]  ，每个线程块大小BM BN,  A矩阵的子块为BM*BK 然而其整行为BM*K
//    for (int k = 0; k < K; k += BK) {
//        //此处看ppt绘图中绿色那块， 这里应该存储的是绿色的一部分
//        //看代码不是 存的是BM*BK  BK*BN这么大的
//        //那就要定位 存哪个块？
//        //看ppt图，线程块（bx,by）负责C[bx][by] ，
//        //每个线程搬运一个元素，总共搬运一套BM*BK  BK*BN，
//        //首先现在 全局的A  B 已经指到正确位置了，  现在只需要管ty tx行了，
//        //擦了 这里BM BN　BK一样大 ，
//        As[ty*BK+tx] = A[ty*K+tx];
//        Bs[ty*BN+tx] = B[ty*N+tx];
//        __syncthreads();//确保一个线程块中的所有线程都搬完了才往下之习性
//        A += BK;
//        B += BK*N;//这里就是指针挪动多少个单位
//        for (int i = 0; i < BK; i++) {
//            tmp += As[ty*BK+i]*Bs[tx+i*BN];  //A子块的ty这行  B子块的tx这列参与计算
//        }
//        __syncthreads(); //等所有线程都算完这里
//    }
//    C[ty*N+tx] = alpha*tmp + beta*C[ty*N+tx];
//}

__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,float beta, float* C) {
    //首先明确 bx by负责计算C[by][bx],每个C分块大小为BM*BN
    int bx = blockIdx.x;
    int by = blockIdx.y;
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    int tx = threadIdx.x % BN;
    int ty = threadIdx.y / BN;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BN*BK];

    //要首先定位到A  B  C的起点，为什么呢？
    //这里要指向各自的起点
    A=&A[by*K*BM];
    B=&B[bx*BN];
    C=&C[by*N*BM+bx*BN];
    float tmp = 0.0f;
    for(int k=0;k<K;k+=BK){
        //写到这要考虑 什么呢？ 要考虑数据搬运了
        As[ty*BK+tx]  = A[ty*K+bx];
        Bs[ty*BN+tx]  = B[ty*N+tx];
        __syncthreads(); //等所有线程都搬运完
        A+=BK;
        B+=BK*N;
        for(int i=0;i<BK;i++){
            tmp += As[ty*BK+i]*Bs[tx+i*BN];
        }
    }
//    C[ty*N+tx] = alpha*tmp + beta*C[ty*N+tx];
//    我感觉这行应该是
    C[ty*BN+tx] = alpha*tmp + beta*C[ty*BN+tx]; //因为已经指向分块起始地址了
}
template<const int BM,const int BN, const int BK, const int TM, const int TN>
__global__ void  mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,float beta, float* C) {
    //首先明确 bx by负责计算C[by][bx],每个C分块大小为BM*BN
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN/TN; //横向有多少tile
    int block_col_thread = BM/TM; //我咋感觉 这命名有点反了呢？
    int thread_num = block_row_thread*block_col_thread;

    int tx = (threadIdx.x % block_row_thread )*TN;
    int ty = (threadIdx.x % block_col_thread )*TM;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BN*BK];  //上边倒是平常 容易理解
    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N+bx*BN];   //这里还是同样理解，一个线程块负责搬运一个C[by][bx],  负责搬运A的一行 ， B的一列


    //下边该如何搬运呢？ 搬运是所有线程都去搬，然后每个数据每次先搬运一个， 循环a_tile_stride次数  此处要看第3页ppt
    int a_tile_row  = threadIdx.x / BK; //相当于搬运A的子块的 第几行
    int a_tile_col =  threadIdx.x % BK; //相当于搬运A的子块的 第几列
    int a_tile_stride = thread_num / BK;  //表示循环几次

    int b_tile_row  = threadIdx.x / BN; //相当于搬运B的子块的 第几行
    int b_tile_col =  threadIdx.x % BN; //相当于搬运B的子块的 第几列
    int b_tile_stride = thread_num / BN;  //表示循环几次

    float tmp[TM][TN] = {0.0f}; //这里还真不明白了 ，为什么这里索引是TM TN？难道这里是个声明了TM TN的二维数组？ 确实是二维数组，每个元素初始化为0.0f

    //这里注意看图2，还是算一个黄色块，所以要循环K次搬运计算
    for(int k=0;k<K;k+=BK){
        //开始搬运子块  如何定位呢？我觉得用 threadIdx.x就可以啊？
//        for (int i =0; i<BM; i+=a_tile_stride){ //此时的i为第几次搬运
//            As[threadIdx.x + i * a_tile_stride*BK] = A[threadIdx.x + i* a_tile_stride *BK];
//        }
//        for (int i =0; i<BN; i+=b_tile_stride){ //此时的i为第几次搬运
//            Bs[threadIdx.x + i * b_tile_stride*BN] = B[threadIdx.x + i* b_tile_stride *BN];
//
//        }
    //既然chatgpt说上述不行， 那只能用a_tile_row  a_tile_col了
        for(int i=0;i<BM;i+=a_tile_stride){
            //这里这个循环 相当于i就是a_tile_stride了
            //右边为什么是K呢？可以看第三页ppt 那是不是说A矩阵指向 &A[by*BM*K]为起点的分块，就是为了a_tile_col这个索引。对应该说基本正确
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row+i)*K + a_tile_col];
        }
        for(int i=0;i<BN;i+=b_tile_stride){
            Bs[(b_tile_row+ i)*BN  + b_tile_col] =  B[(b_tile_row+i)*N + b_tile_col];
        }

        }
        __syncthreads();
        A+= BK;
        B+=BK*N; //这个挪的是黄色那个的位置

        //这里开始计算  此时又得看第二张PPT了  盯了一会 那红色的四个格子的tmp是累加而来的
        //那么改如何定位到As  Bs中呢？  突然想起来 知道 ty tx了 即 知道在C中的起始地址了，那么可以根据 TM  TN的索引算一下呗
        //开始索引是ty tx
//        for(int j=0;j<TM;j++){
//            for(int i=0;i<TN;i++){
//                for(int k=0;k<BK;k++){
//                    tmp[j][i] +=  As[(ty+j)*BK+k]*Bs[tx+i + k*BN];
//                }
//            }
//        }
        for(int i=0;i<BK;i++){  //这一层是干什么的来？ 遍历BK的宽度  乘积和嘛
            for(int j=0;j<TM;j++){
                for(int l =0; l<TN ; l++){
                    tmp[j][l] += As[(ty+j)*BK+i]*Bs[tx+l+i*BN];
                }
            }
        }
        __syncthreads();
    //把tmp结果 写回对应的位置
    for(int j=0;j<TM;j++){
        for(int l=0;l<TN;l++){
            C[(ty+j)*N+tx+l] = alpha*tmp[j][l] + beta*C[(ty+j)*N+tx+l];
        }
    }

    }
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main()
{
    std::vector<int> sizes = {128,256,512,1024,2048,4096,8192};
    std::ofstream csv_file("/cuda_code/tmp/sgemm_benchmark_v3.csv");

    for ( auto N :sizes)
    {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = N*N*sizeof(float);
        float *A = (float *)malloc(size);
        float *B = (float *)malloc(size);
        float *C_cublas = (float *)malloc(size);
        float *C_v1 = (float *)malloc(size);
        float *d_A,*d_B,*d_C_v1;
        checkCudaError(cudaMalloc(&d_A,size),"cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B,size),"cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1,size),"cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;
        try{
            for (int i = 0; i < N*N; i++) {
                A[i] = 1.0f;
                B[i] = 2.0f;
            }
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy B to device failed");
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // warmup
            int warpup_time = 10;  // 热身次数
            for (int i = 0; i < warpup_time; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                             &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();
            // cuBLAS SGEMM
            int repeat_time = 5;
            checkCudaError(cudaEventRecord(start),
                           "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                             &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }

            checkCudaError(cudaEventRecord(stop),
                           "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop),
                           "cudaEventSynchronize cublas failed");

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // 拷贝 cuBLAS 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_cublas failed");

            // mysgemm_v1
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
//            dim3 threads(BLOCK_SIZE, BLOCK_SIZE);//此处BLOCK_SIZE为32 cuda中没有Dim2这个类型 ，实际是（BS,BS,1） 。这一行 只是定义了 block 的内部线程布局，
//            dim3 blocks((N + threads.x - 1) / threads.x,
//                        (N + threads.y - 1) / threads.y);
//            for (int i = 0; i < warpup_time; ++i) {
//                mysgemm_v1<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
//            }
//            cudaDeviceSynchronize();
//            checkCudaError(cudaEventRecord(start),
//                           "cudaEventRecord(start v1) failed");
//            for (int i = 0; i < repeat_time; ++i) {
//                mysgemm_v1<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
//            }


            //mysgemm_v2==========================
//            dim3 blockDim(1024);
//            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));
//
//
//            for (int i = 0; i < warpup_time; ++i) {
//                mysgemm_v2
//                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
//            }
//
//            cudaDeviceSynchronize();
//            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
//
//            checkCudaError(cudaEventRecord(start),
//                           "cudaEventRecord(start v1) failed");
//
//            for (int i = 0; i < repeat_time; ++i) {
//                mysgemm_v2
//                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
//            }
//

            //mysgemm_v2===========================


            //mysgemm_v4===========================

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i) {
                mysgemm_v4<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }


            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start),
                           "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i) {
                mysgemm_v4<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }


            //mysgemm_v4===========================





            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop),
                           "cudaEventSynchronize v1 failed");

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime v1 failed");

            // 拷贝手写 kernel 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_v1 failed");
            // 结果比较
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i) {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
                    error_count++;
                }
            }
            float cublas_gflops =repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
            float v1_gflops =repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v1);

            free(A);
            free(B);
            free(C_cublas);
            free(C_v1);
        }
        catch (...){
            std::cerr << "Out of memory or error during testing size: " << N
                      << std::endl;
            out_of_memory = true;
        }
        if (!out_of_memory) {
            std::cout << "Finished size: " << N << std::endl;
        } else {
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }
    csv_file.close();

//    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
//              << std::endl;
    return 0;
}