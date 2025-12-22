#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void checkCudaError(cudaError_t err,const char *msg){
    if(err != cudaSuccess){
        std::cerr<< msg << "CUDA ERROR: " << cudaGetErrorString(err)<<std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t err,const char *msg){
    if(err != CUBLAS_STATUS_SUCCESS){
        std::cerr<< msg << "CUBLAS ERROR: " << err<<std::endl;
        exit(EXIT_FAILURE);
    }
}

template<const int BM, const int BN, const int BK,const int TM, const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C){
    //确定块索引
    int bx = blockIdx.x ;
    int by = blockIdx.y ;

    int block_row_thread = BN / TN ; //C子块 行方向需要多少线程
    int block_col_thread = BM / TM ; //C子块  列方向需要多少线程
    int thread_num = block_row_thread * block_col_thread ; //每个block需要多少线程

    //确定每个线程处理的tile的起始 列 行索引
    int tx = (threadIdx.x % block_col_thread) * TN ;
    int ty = (threadIdx.x / block_col_thread) * TM ;

    __shared__ float As[BK *BM]; //注意这里As是转置的
    __shared__ float Bs[BK *BN];

    // 一行需要（ BK/4）个线程搬运，那么总共需要 BM*（BK/4）个线程次，  线程次除以线程总数 即为搬运次数
    const int ldg_a_num = BM*(BK/4) / thread_num ;
    const int ldg_b_num = BN*(BK/4) / thread_num ;

    //现在要算每个线程要办字块的地价和那个 第几列
    int a_tile_row = threadIdx.x / (BK/4);
    int a_tile_col = threadIdx.x % (BK/4) * 4;
    int a_tile_stride = BM / ldg_a_num; //每个线程下次  跨的行数

    int b_tile_row = threadIdx.x / (BN/4);
    int b_tile_col = threadIdx.x % (BN/4) * 4;
    int b_tile_stride = BK / ldg_b_num; //

    float accum[TM][TN] = {0.};//每个线程在寄存器里开辟的本地累加数组 用来存储线程负责的TM * TN 小块的最终计算结果。

    float ldg_a_reg[4* ldg_a_num] = {0.};

    float a_frag[TM];
    float b_frag[TN];

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM + bx*BN];

    for(int k= 0 ; k<K; k+=BK){
        //开始搬运 计算
        for(int i =0; i<BM ;i+=a_tile_stride){
            int ldg_index =  i/a_tile_stride *4 ;
            //这么写  也就搬运一次啊
            //实际搬运多次， i+a_tile_row 是实际的 搬运的行数
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row,a_tile_col,K)]);
            As[OFFSET(a_tile_col, a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col+1, a_tile_row+i, BM)] = ldg_a_reg[ldg_index+1];
            As[OFFSET(a_tile_col+2, a_tile_row+i, BM)] = ldg_a_reg[ldg_index+2];
            As[OFFSET(a_tile_col+3, a_tile_row+i, BM)] = ldg_a_reg[ldg_index+3];
        }
        for(int i =0; i<BN ;i+=b_tile_stride){
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row+i, b_tile_col, BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row+i, b_tile_col, N)]);
        }
        __syncthreads();
        A+=BK;
        B+=BK*N;

        //搬完了 开始算吧
        for(int i =0; i<BK; i++){
            for(int m=0;m<TM; m+=4){
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i,ty+ m, BM)]);
            }
            for(int n=0;n<TN; n+=4){
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(ty+n, i, BN)]);
            }
            for(int m=0;m<TM; m++){
                for(int n=0;n<TN; n++){
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();

        //结果写回  结果写回也是float4写回
        for(int m=0;m<TM; m++){
            for(int n=0;n<TN; n+=4){
                float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty+m, ty+n, N)]);
                ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
                ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
                ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
                ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
                FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
            }
        }

    }

}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

std::vector<int> generateSizes() {
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256) {
        sizes.push_back(i);
    }
    return sizes;
}

int main(){
    std::vector<int> sizes = generateSizes();
    std::ofstream csv_file("/cuda_code/tmp/sgemm_benchmark_v4.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;
    for(int N :sizes){
        std::cout << "Size: " << N << std::endl;
        size_t size = N*N * sizeof(float);
        float *A = (float *) malloc(size);
        float *B = (float *) malloc(size);
        float *C_cublas = (float *) malloc(size);
        float *C_v1 = (float *) malloc(size);

        float *d_A , *d_B , *d_C_v1;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C_v1, size);

        bool out_of_memory = false;

        try{
            for(int i = 0; i < N*N; i++){
                A[i] = 1;
                B[i] = 2;
            }
            cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle),"cublasCreate failed");

            float alpha = 1.0f, beta = 0.0f;

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int warmup_time = 10;
            for(int i = 0; i < warmup_time; i++){
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C_v1, N),"cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            int repeat_time =5;
            cudaEventRecord(start);
            for(int i = 0; i < repeat_time; i++){
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C_v1, N),"cublasSgemm failed");
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float cublas_time = 0;
            cudaEventElapsedTime(&cublas_time, start, stop);
            cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost);

            cudaMemset(d_C_v1, 0, size);

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warmup_time; ++i) {
                mysgemm_v6<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();
            cudaMemset(d_C_v1, 0, size);
            cudaEventRecord(start);
            for (int i = 0; i < repeat_time; ++i) {
                mysgemm_v6<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime v1 failed");

            // 拷贝手写 kernel 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_v1 failed");


        }





















}
