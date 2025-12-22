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
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v6(int M,int N, int K, float alpha,  float *A,  float *B, float beta, float *C){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num =  block_row_thread * block_col_thread;

    int ty = (threadIdx.x / block_row_thread) *TM;
    int tx =  (threadIdx.x % block_row_thread) *TN;

    __shared__ float As[BK * BM]; //注意此处不是BM *BK了
    __shared__ float Bs[BK * BN];

    //现在求A B的子块搬运到As Bs的次数; 对于
    const int ldg_a_num = BM /( thread_num/(BK/4)) ; //一行BK需要 （Bk/4）个线程搬
    const int ldg_b_num = BN /( thread_num/(BK/4));

    //现在要计算的是 每个线程负责搬运的索引的行号 列号
    int a_tile_row =  threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    //求A子块 搬运的幅度 ， 就是搬运一次是多少行
    int a_tile_stride = BM / ldg_a_num;


    //同样算B的
    int b_tile_row =  threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BN / ldg_b_num;

    float accum[TM][TN] = {0.0f};//存储结果
    float ldg_a_reg[4 * ldg_a_num] = {0.};//存储A的一个float4， 为什么要这么大呢？哦 ldg_a_num是搬运次数，

    float a_frag[TM];
    float b_frag[TN];

    //写起始地址还是得看第一页PPT的
    A = &A[by * BM * K];
    B = &B[bx*BN];
    C= &C[by * BM*N+ bx*BN]; //C矩阵的起始地址

    for(int k =0; k<K; k+=BK){ //依旧稳定发挥 因为A的一行 B的一列有这么多块参与计算

        //搬运A子块到共享内存  这里为什么不用ldg_a_num 当次数呢  已经算出来搬运次数了不是？
        for (int i = 0 ; i<BM ; i += a_tile_stride){
            int ldg_index = i / a_tile_stride * 4; //一个float4 在寄存器缓存中 存储的 起始位置

            //FETCH_FLOAT4是一个放入函数 只需写上首地址，其放入四个float
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            //实际上，上边这行等于
//            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[(a_tile_row + i)*K + a_tile_col]);  //注意此处的每行是K
            As[OFFSET(a_tile_col,i+a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col+1,i+a_tile_row, BM)] = ldg_a_reg[ldg_index+1];
            As[OFFSET(a_tile_col+2,i+a_tile_row, BM)] = ldg_a_reg[ldg_index+2];
            As[OFFSET(a_tile_col+3,i+a_tile_row, BM)] = ldg_a_reg[ldg_index+3];
        }

        //搬运B子块到共享内存
        for (int i = 0 ; i<BK ; i += b_tile_stride){
            //b_tile_row  + i 是相当于真正的行号
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row+i ,b_tile_col,BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row+i,b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;
        B += BK * N;


        //此时已经挪完数据 需要明确的是As中存放的是原先A输入矩阵的转置  可以开始计算了

    }
}
