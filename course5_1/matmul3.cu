#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
//行号 列号  列大小
#define OFFSET(row, col, ld) ((row) * (ld) + (col)) //把二维坐标转换为一维数组索引



//float4 是 CUDA 提供的 4 元浮点向量类型，类似 (x, y, z, w)。
//reinterpret_cast<float4 *>(&(pointer)) → 把 pointer 的地址解释成 float4* 类型
//[0] → 取这个 float4 的第 0 个元素（也就是这 4 个 float 的整体）

//假设 A 是一维数组 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ...]
//FETCH_FLOAT4(A[0]) → 拿到 (1.0, 2.0, 3.0, 4.0)，相当于一次搬运 4 个桌子。
//pointer 实际上是 一个数组元素（比如 ldg_a_reg[ldg_index]）。不是指针
//地址还有类型，因为编译器需要知道这个地址指向什么类型的数据，才能正确计算偏移量和访问内存。
//为什么不用static_cast:static_cast
//只能做相关类型之间的转换（比如 int 转 float，或者类的父子类型转换）。
//不能用来把 float* 转 float4*，因为它认为这俩不兼容。
//这里 [0] 等价于 * 解引用：
//reinterpret_cast<float4 *>(&(pointer)) → float4* 类型
//[0] 取第 0 个元素，相当于 *，即解引用，得到 float4。
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

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

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN; //看图  该值表示C子块  每行需要多少线程处理
    const int block_col_thread = BM / TM; //每列 tile 需要多少线程。
    const int thread_num = block_row_thread * block_col_thread; //小教室（线程块）中的线程总数

    int tx = (threadIdx.x % block_row_thread) * TN; //每个线程处理的 tile 的起始列索引（桌子左上角的列）。
    int ty = (threadIdx.x / block_row_thread) * TM; //每个线程处理的 tile 的起始行索引（桌子左上角的行）。

    __shared__ float As[BK * BM]; // A子矩阵的共享内存  是小教室里 A 的讲台
    __shared__ float Bs[BK * BN]; //B子矩阵的共享内存

    const int ldg_a_num = BK * BM / thread_num / 4; //每个线程需要从全局内存加载的 float4 数量
    const int ldg_b_num = BK * BN / thread_num / 4;

    //threadIdx是一维的 但是要负责填充二维BM*BK子块，必须把它们映射成二维任务分配，每个线程知道自己处理哪一行哪一列
    int a_tile_row = threadIdx.x / (BK / 4);//线程在共享内存 A 子块中负责的 起始行, (BK / 4)代表每行需要多少个线程来处理

    //threadIdx.x % (BK / 4)代表线程在当前行的编号
    int a_tile_col = threadIdx.x % (BK / 4) * 4; //线程在共享内存 A 子块中负责的 起始列,threadIdx.x % (BK / 4) 表示线程在这一行是第几个

    //同一线程下次再处理下一行时的行数（跨行）
    //ldg_a_num  每个线程一次可以加载多少份 float4 数据
    //假设参数
    //BM = 12 行（A 子块高 12 行）
    //BK = 8 列（A 子块宽 8 列）
    //线程块内线程数 threadNum = 6
    //每次搬 4 个元素（float4）
    //计算每个线程负责多少桌子
    //总桌子数 = BM * BK = 12 * 8 = 96
    //每个线负责 = 96 / 6 = 16 个桌子
    //每次搬 4 个 → 16 / 4 = 4 次
    //✅ 所以 ldg_a_num = 4
    int a_tile_stride = BM / ldg_a_num; //按照上述原理 感觉不应该这么算 这应该是个化简的形式

    int b_tile_row = threadIdx.x / (BN / 4);//线程在共享内存B子块中的起始行，（BN/4）表示每行需要多少线程来处理
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;

    float accum[TM][TN] = {0.};//每个线程在寄存器里开辟的本地累加数组 用来存储线程负责的TM * TN 小块的最终计算结果。

    float ldg_a_reg[4 * ldg_a_num] = {0.}; //每个线程一次从全局内存搬运的 A矩阵元素缓存（float4）

    float a_frag[TM];//每次计算前，从共享内存里读取 一行或一列的数据片段到寄存器
    float b_frag[TN];


    //每个小教室（线程块）对应矩阵的一个子块   int bx = blockIdx.x;     int by = blockIdx.y;
    // bx* BN ->小教室在大教室横向的偏移   //by* BM -> 小教室在纵向的偏移 ；
    //小教室的位置确定了 学生负责哪一排哪一列的桌子
    // BM  BN 分别是一个C中一个小教室（tile）的大小
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN]; //在C矩阵的起始地址

#pragma unroll
    //外层 for (k...) 是按 K 维度把 A 分成宽度为 BK 的片段；
    //我这里不理解 对矩阵A来说 一个线程块不是负责 一个 BM*BK吗，为啥还要外层循环呢？
    //每个线程块计算 C 的一块（BM×BN），这个 C 子块依赖 A 和 B 的所有 K 维度数据 ,结合图来理解
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        //内层 是每个线程把自己负责的几组float4(4个float)从全局内存A中读出，写入共享内存As的正确位置，一边后面所有线程共享专用
        //遍历A子块的行 ， 步长记为a_tile_stride （a_tile_stride 表示同一个线程在循环中再次加载 A 子块元素时，沿行方向要跳过多少行）
        //假如BM=128   a_tile_stride=32  则每个线程负责4次搬运
        //第一次搬运 i=0 ,实际行 =a_tile_row + i; 第二次：i = 32 → 实际行 = a_tile_row + 32 ;
        //这样所有线程协作，就能均匀覆盖 128 行。
        //假设：
        //a_tile_stride = 3
        //i 的值依次是 0, 3, 6, 9（共 4 次）
        //那么：
        //i=0 → i / a_tile_stride = 0 → ldg_index = 0*4 = 0
        //i=3 → i / a_tile_stride = 1 → ldg_index = 1*4 = 4
        //i=6 → i / a_tile_stride = 2 → ldg_index = 2*4 = 8
        //i=9 → i / a_tile_stride = 3 → ldg_index = 3*4 = 12
        for (int i = 0; i < BM; i += a_tile_stride) {
            //(i / a_tile_stride) 表示 第几次搬运（从 0 开始）
            //对应的寄存器的区间
            //第1次 → ldg_a_reg[0..3]
            //第2次 → ldg_a_reg[4..7]
            //第3次 → ldg_a_reg[8..11]
            //第4次 → ldg_a_reg[12..15]
            int ldg_index = i / a_tile_stride * 4; //计算当前线程在自己的寄存器缓存ldg_a_reg中存储位置的索引，因为每次搬运4个浮点数

            //这几块代码是有分工的，但逻辑上属于一个“搬运流程”，先从全局内存搬到寄存器，再从寄存器写到共享内存。原因是 CUDA 不允许直接用 float4 从全局内存加载后立刻写入共享内存，需要先放到寄存器。
            //从全局内存的 A 中搬运 4 个连续元素到寄存器 ldg_a_reg。
            //这句相当于
            //(reinterpret_cast<float4 *>(&(ldg_a_reg[ldg_index]))[0]) =(reinterpret_cast<float4 *>(&(A[OFFSET(a_tile_row + i, a_tile_col, K)]))[0]);
            //A[OFFSET(...)]是A矩阵全局内存的某个元素，同样，取它的地址，reinterpret_cast 成 float4*，再 [0] 取出值。
            //a_tile_row：这个线程在共享内存 A 子块中负责的 起始行索引（基于线程 ID 计算）。
            //i 外层循环变量，表示这个线程 加载的第几次行偏移（因为每个线程会跨行加载多次数据）。

            //i = 0 → i / a_tile_stride = 0 → ldg_index = 0*4 = 0
            //        从 A 加载：A[(a_tile_row+i)=2 行, 列 0~3] → A[2][0], A[2][1], A[2][2], A[2][3]
            //        存入寄存器：ldg_a_reg[0], ldg_a_reg[1], ldg_a_reg[2], ldg_a_reg[3]
            //
            //i = 3 → i / a_tile_stride = 1 → ldg_index = 1*4 = 4
            //        从 A 加载：A[(a_tile_row+i)=5 行, 列 0~3] → A[5][0], A[5][1], A[5][2], A[5][3]
            //        存入寄存器：ldg_a_reg[4], ldg_a_reg[5], ldg_a_reg[6], ldg_a_reg[7]
            //
            //i = 6 → i / a_tile_stride = 2 → ldg_index = 2*4 = 8
            //        从 A 加载：A[(a_tile_row+i)=8 行, 列 0~3] → A[8][0], A[8][1], A[8][2], A[8][3]
            //        存入寄存器：ldg_a_reg[8], ldg_a_reg[9], ldg_a_reg[10], ldg_a_reg[11]
            //
            //i = 9 → i / a_tile_stride = 3 → ldg_index = 3*4 = 12
            //        从 A 加载：A[(a_tile_row+i)=11行, 列 0~3] → A[11][0], A[11][1], A[11][2], A[11][3]
            //        存入寄存器：ldg_a_reg[12], ldg_a_reg[13], ldg_a_reg[14], ldg_a_reg[15]
            //右边的 FETCH_FLOAT4 会把这个 float 地址 reinterpret_cast 成 float4*，一次性取出 4 个 float。
            //左边ldg_a_reg[ldg_index] 是 float 的起始位置，左边也用 FETCH_FLOAT4，是告诉编译器：“我要把右边的 float4 整个拷贝到左边连续 4 个 float 中去”
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);

            //把 寄存器中的数据（ldg_a_reg）写入共享内存 As，并且按正确位置存储，确保 后续所有线程可以共享这部分数据。
            //i = 0 → ldg_index = 0
            //        写入共享内存：
            //        As[ (col=0, row=2) ] = ldg_a_reg[0]
            //        As[ (col=1, row=2) ] = ldg_a_reg[1]
            //        As[ (col=2, row=2) ] = ldg_a_reg[2]
            //        As[ (col=3, row=2) ] = ldg_a_reg[3]
            //
            //i = 3 → ldg_index = 4
            //        写入共享内存：
            //        As[ (col=0, row=5) ] = ldg_a_reg[4]
            //        As[ (col=1, row=5) ] = ldg_a_reg[5]
            //        As[ (col=2, row=5) ] = ldg_a_reg[6]
            //        As[ (col=3, row=5) ] = ldg_a_reg[7]
            //
            //i = 6 → ldg_index = 8
            //        写入共享内存：
            //        As[ (col=0, row=8) ] = ldg_a_reg[8]
            //        As[ (col=1, row=8) ] = ldg_a_reg[9]
            //        As[ (col=2, row=8) ] = ldg_a_reg[10]
            //        As[ (col=3, row=8) ] = ldg_a_reg[11]
            //
            //i = 9 → ldg_index = 12
            //        写入共享内存：
            //        As[ (col=0, row=11) ] = ldg_a_reg[12]
            //        As[ (col=1, row=11) ] = ldg_a_reg[13]
            //        As[ (col=2, row=11) ] = ldg_a_reg[14]
            //        As[ (col=3, row=11) ] = ldg_a_reg[15]
            //为什么写入共享内存 行不是从 0 开始，而是从 2、5 开始
            //为什么共享内存访问是 As[col, row]，列在前，而不是行在前
            //第一次搬运放到 2 行，第二次搬运放到 5 行（i=3 + a_tile_row=2）
            //原因：
            //每个线程只负责矩阵子块的一部分行，不是从 0 开始填整个共享内存。
            //如果线程 ID 是 0，它可能负责行 0~3
            //线程 ID 是 1，它可能负责行 4~7
            //所以 a_tile_row 决定了 线程负责的起始行
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
#pragma unroll
        //将全局内存 B 的一块子矩阵（B 子块）搬运到共享内存 Bs。
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();//同步线程：确保所有线程都完成搬运，共享内存 Bs 准备好，才能进行下一步矩阵乘。
        A += BK; //每次循环后，A、B 指向下一块 K 维度的全局内存子矩阵。 哦这个循环是104行那个
        B += BK * N;
#pragma unroll
        //假设参数
        //BM = 12，BK = 8
        //threadNum = 6
        //每次搬 4 个 float → 总共每线程 16 个元素 → 4 次搬运
        //TM = 2，TN = 2 → 每个线程负责 2×2 C 子块
        //ty / tx → 当前线程在子块的行/列偏移（假设从 0 开始）
        //a_frag[TM] / b_frag[TN] → 寄存器缓存
        //accum[TM][TN] → 寄存器累加结果
        for (int i = 0; i < BK; i++) {
#pragma unroll
            // 加载 A 到寄存器 a_frag
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]);//从 A 的共享内存子块 As 中读取 当前 K 维的第 i 行，搬到寄存器 a_frag[m]。
            }
#pragma unroll
            //  // 2. 加载 B 到寄存器 b_frag
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]);//从 B 的共享内存子块 Bs 中读取当前 K 维的第 i 行（对应 C 子块的列方向）。
            }
#pragma unroll
            // 3. 寄存器累加 TM×TN 子块
            for (int m = 0; m < TM; m++) {//在寄存器里计算 TM×TN 小 tile 的乘加结果。accum[m][n] 是线程自己的累加结果。
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();//所有线程同步，确保共享内存 As 和 Bs 数据都加载完成后，才进入下一轮 K 维迭代
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]); //将更新后的4个元素写回全局内存C
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
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

int main() {
    std::vector<int> sizes = generateSizes();

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v4.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N: sizes) {
        std::cout << "Testing size: " << N << std::endl;

        size_t size = N * N * sizeof(float);
        float *A = (float *) malloc(size);
        float *B = (float *) malloc(size);
        float *C_cublas = (float *) malloc(size);
        float *C_v1 = (float *) malloc(size);

        float *d_A, *d_B, *d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;

        try {
            // 初始化矩阵 A 和 B
            for (int i = 0; i < N * N; ++i) {
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            // 拷贝到设备
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

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i) {
                mysgemm_v6<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start),
                           "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i) {
                mysgemm_v6<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
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

            float cublas_gflops =
                    repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
            float v1_gflops =
                    repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
            // 写入CSV
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

        } catch (...) {
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

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
              << std::endl;
    return 0;
}
