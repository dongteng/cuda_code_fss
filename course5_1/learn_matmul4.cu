#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define BLOCK_SIZE 128
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

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256) //每个thread block的线程数上限为256
mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];
    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;

    float accum[TM][TN] = {0.};

    float ldg_a_reg[4 * ldg_a_num] = {0.};
    float ldg_b_reg[4 * ldg_b_num] = {0.};//之前没有b这个寄存器啊

    float a_frag[2][TM];
    float b_frag[2][TN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    //1. 也就是说先加载第 0 个 tile，
    //2. 然后进入循环：
    //2.1 预取下一块数据到另一块 共享内存
    //2.2 开算本块（期间伴随着小流水的切换 ）
    //3 写回

    //------------------------- 大块初始化准备--------------------------------
    //先写第0块 到共享显存As[0] Bs[0]  这里还是看 7 8 页PPT 才写索引
    for (int i = 0; i < BM; i += a_tile_stride) {
        // 哦这里居然自己写成这样？ 应该是不对的
        // 因为自己这样算的是每个线程 在寄存器中的index， 实际上应该算的是  一个线程搬运第几次，也就是在寄存器中的index
//        int ldg_index =  threadIdx.x / (BK / 4);
        int ldg_index = i / a_tile_stride * 4; //这里应该是算第几次搬运，也就是在寄存器中的index
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        As[0][OFFSET(a_tile_col, a_tile_row + i, BK)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, a_tile_row + i, BK)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, a_tile_row + i, BK)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, a_tile_row + i, BK)] = ldg_a_reg[ldg_index + 3];
    }
    for (int i = 0; i < BK; i += b_tile_stride) { //注意此时的限制是BK  看PPT
//        int ldg_index = i/b_tile_stride *4 ; //这里不要这玩意啊 直接写入啊
//        ldg_b_reg[ldg_index] = B[OFFSET(a_tile_row+i,a_tile_col,N)];
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(
                B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }
    __syncthreads(); //这里为什么要同步？是因为写这个块的时候是好多线程合作的

    //-----------------------小流水初始化准备  先搬运一次去 计算寄存器  看P11--------
    for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]);//a_frag
    }
    for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, ty + n, BN)]);//b_frag
    }
    //至此 初始化完成  开始循环

    int write_index = 1; //进入循环要使用的块索引
    int load_index;
    int k = 0;

    do {
        //进入大循环 预取下一块到另一块共享内存
        //计算本块  期间伴随着小流水的切换
        k += BK;
        if (k < K) {
            //预取下一块到另一块共享内存
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(
                        A[OFFSET(a_tile_row + i, a_tile_col + k, K)]);//这里不对吧？只放到临时寄存器里去？没放到共享显存啊？
            }
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(
                        B[OFFSET(a_tile_row + i, a_tile_col + k, N)]);//这里很正确啊 是a_tile_col+k 也是跟其他不一样的地方
            }
            //上边也没有搬运到共享显存啊？？？ 哦 好像是每个线程都有一个ldg_a_reg  ldg_b_reg  先存在这里
            load_index = write_index ^ 1; //对load_index取反

            //开算第一大块！ 期间伴随小流水！  这里也不能说第一大块
            for (int bk = 0; bk < BK - 1; bk++) {  //这里为啥是BK-1呢？
                //写到下面 猜测是因为 小流水中已经有一个数据了，所以从bk+1开始，也就是实际上不从0开始，从1开始
                //一进来 先搬运下一个小流水所需的数据

                //-------------先搬运下一个小流水所需数据
                for (int m = 0; m < TM; m += 4) {
                    //哦 这里开始小流水了 要搬运 a_frag[0、1] b_frag[0、1]  ,load_index  write_index控制的是大流水的索引
                    //为什么要从bk+1开始呢？因为进循环之前小流水中 也就是[0]索引已经有了 。
                    FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(As[load_index][OFFSET(bk + 1, ty + m, BM)]);
                }
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(Bs[load_index][OFFSET(bk + 1, ty + n, BN)]);
                }


                //------------上边 m n是搬运了下一个的小流水
                //------------下边是计算 本次的小流水
                for (int m = 0; m < TM; m++) {
                    for (int n = 0; n < TN; n++) {
                        accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                    }
                }
            }
            //上边其实就算完了？  没算完  因为小流水的最后一轮还没有算  上边的循环只到BK-2的  Bk-1索引那个小流水还没算呢
            //下边就是 把大流水搬运完全，因为大流水只搬了一半 没放进共享矩阵里去
            if (k < K) {
                //将在临时寄存器中的数据 按照转置的顺序写到共享矩阵
                //写这块可以参照 P17
                for (int i = 0; i < BM; i += a_tile_stride) {
                    int ldg_index = i / a_tile_stride * 4;
                    As[write_index][OFFSET(a_tile_col, a_tile_row + i, BM)] = ldg_a_reg[ldg_index];
                    As[write_index][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 1];
                    As[write_index][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 2];
                    As[write_index][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 3];
                }
                //注意B不需要 转置存储
                for (int i = 0; i < BK; i += b_tile_stride) {
                    int ldg_index = i / b_tile_stride * 4;
                    Bs[write_index][OFFSET(b_tile_col, b_tile_row + i, BN)] = ldg_b_reg[ldg_index];
                }
                __syncthreads();

                //上边是把下一个大流水  完整的搬运结束
                //下边给下一大轮的小流水准备一个
                //为什么是 小流水是写到 0呢？！  必然的呀  只有2个  而且load_index  write_index是控制大块索引的
                for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[load_index][OFFSET(0, ty + m, BM)]);
                }
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[load_index][OFFSET(0, ty + n, BN)]);
                }
                write_index = load_index ^ 1; //对write_index取反
            }

            //这段是干什么的？
            // chatgpt说这是小流水的最后一轮计算你的小流水使用：
            //
            //a_frag[0], a_frag[1]
            //b_frag[0], b_frag[1]每次循环做两个动作：
            //预取下一小块到 a_frag[(bk+1)%2]
            //计算当前小块 a_frag[bk%2]
            //所以这个循环里只计算了 bk = 0, 1, ..., BK-2
            //为啥停在BK-2?
            //因为：小流水开始前已经加载了第 0 块到 a_frag[0]   循环内部每次加载的是下一块（bk+1）
//            因此：
//            当循环跑到 bk = BK-2 时
//            它预取了 a_frag[(BK-1)%2]
//            但 没有计算最后这个小流水（编号 BK-1）
//            所以循环跑到 bk = BK-2 时就结束了
//            最后一个小流水BK-1在循环外部计算,那么它存在哪个槽里？
//            当bk=Bk-2时，（循环最后一次迭代）
//预取的是：bk + 1 = BK - 1
//槽号  = (BK - 1) % 2
//因此最后的 小块被放在（(BK - 1) % 2）里
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
                }
            }


        }
    } while (k < K);

    //结果写回
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, ty + n, N)]);
            ctmp.x += accum[m][n] * alpha + beta * ctmp.x;
            ctmp.y += accum[m][n] * alpha + beta * ctmp.y;
            ctmp.z += accum[m][n] * alpha + beta * ctmp.z;
            ctmp.w += accum[m][n] * alpha + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, ty + n, N)]) = ctmp;
        }
    }


}