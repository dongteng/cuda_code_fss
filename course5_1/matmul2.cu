#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

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

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //首先 计算了一个大小为BM*BK的矩阵分块所需要的线程总数，记为thread_num
    //举例 大教室-矩阵C整体，小教室-一个线程块（负责BM*BN的矩阵），桌子（矩阵里的元素），学生（一个线程）学生分配

    int block_row_thread = BN / TN; // 横向有多少个 tile
    int block_col_thread = BM / TM; // 纵向有多少个 tile
    int thread_num = block_row_thread * block_col_thread;


    //们将线程块中的一维线程索引展开为二维形式，从而得到线程在 x 轴和 y 轴上的局部索引 tx 和 ty.假设小教室8*8
    //学生 0：负责 (0,0) 开始的 2×2 桌子
    //学生 1：负责 (0,2) 开始的 2×2 桌子
    //学生 2：负责 (0,4) 开始的 2×2 桌子
    //学生 3：负责 (0,6) 开始的 2×2 桌子
    //学生 4：负责 (2,0) 开始的 2×2 桌子
    //...
    //学生 15：负责 (6,6) 开始的 2×2 桌子
    //对应到代码就是下边两行
    //tx , ty指的是这个线程 负责的这个小教室的左上角的坐标  这里可以看自己化的一张图
    int tx = (threadIdx.x % block_row_thread) * TN; //这里的threadIdx.x是一维的 外部调用是256。threadIdx.x 对 4（tile的列还是行？） 取余，决定它在第几列，然后乘以 TN，算出起始桌子列号。
    int ty = (threadIdx.x / block_row_thread) * TM;//TN 表示一个学生（线程）在水平方向要负责的桌子数。TM 表示一个学生（线程）在竖直方向要负责的桌子数。

    __shared__ float As[BM * BK];//这是 教室里的黑板，提前写好这节课需要的资料（从全局内存里搬进来，大家共享）。
    __shared__ float Bs[BK * BN];

    //牢记 M * K 是矩阵A的大小 ，K*N是矩阵大小，M × N：矩阵 C 的大小。
    //BM × BK：每个线程块处理 A 的一个子块大小。
    //BK × BN：每个线程块处理 B 的一个子块大小。
    //BM × BN：每个线程块负责 C 的一个子块大小
    //bx, by：block 在整个大矩阵里 横向、纵向的编号。

    //类比场景
    //整栋教学楼 = 整个矩阵 C（M×N）。
    //每个教室（block）只负责这栋楼里的一小块桌子（BM×BN）。
    //学生（线程）要知道：
    //我的教室负责大矩阵哪一块？
    //从 A、B 的哪一块拿数据才能算出这块 C？


    //每个 block 在 纵向（M 方向） 负责 BM 行。
    //by * BM = 当前 block 的起始行。
    //乘以 K（因为 A 有 K 列，每行有 K 个元素），就是这个子块在内存中的起始地址。
    A = &A[by * BM * K];//这行代码只是把 指针挪到 A 的第 by*BM 行开头。但整个 block 不只是用这一行，它会在后续循环中取 BM 行、BK 列 的一个子矩阵。后面会用这一行往下 BM 行、往右 BK 列的子矩阵。
    B = &B[bx * BN];//每个 block 在 横向（N 方向） 负责 BN 列。 bx * BN = 当前 block 的起始列。先把指针对准 B 的 起始列，后面会用这一列往下 BK 行、往右 BN 列的子矩阵。

    // M * N是矩阵C的大小
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK; //计算线程在A子块方向上负责哪一行
    int a_tile_col = threadIdx.x % BK; //计算线程在列方向上 负责哪一列tile
    int a_tile_stride = thread_num / BK; //这个代表 一行 需要有几个 线程负责

    int b_tile_row = threadIdx.x / BN; //BN：B 子块在列方向的大小。 线程用这些索引计算自己在 B 子块里负责哪一行哪一列，
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN; //这个代表矩阵 B  的一行要几个线程负责

    //每个线程在寄存器里开辟的局部数组，用来保存自己处理的 tile 的计算结果。TM TN指的是每个线程负责的小块大小
    //学生自己准备一个小笔记本 记录它负责的那几个桌子的最终得分，等最后统一写回
    float tmp[TM][TN] = {0.};
#pragma unroll //相当于把循环展开 不再执行循环了
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        //作用：线程将 A 的一部分从全局内存搬到共享内存 As.学生把 A 教室的桌子搬到共享的过道（shared memory），方便大家取。
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        // 学生也搬 B 教室的桌子到共享过道。
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] =
                    B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        //遍历 BK 维度（加载的 tile），计算累加到 tmp[j][l]。
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int j = 0; j < TM; j++) {
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
            }
        }
        __syncthreads();
    }
#pragma unroll
    //把 tmp 中的结果写回 C 的对应位置。
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] =
                    alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
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
    std::ofstream csv_file("/cuda_code/tmp/sgemm_benchmark_v3.csv");
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

            // mysgemm_v4
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

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
