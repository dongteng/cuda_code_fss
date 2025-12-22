#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <fstream>
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

template<const int BM, const int BN, const int BK, const int row_stride_a,const int row_stride_b>
__device__ void load_from_gmem(int N, int K, const float *A, const float *B,float *As,float *Bs,int inner_row_a,int inner_col_a,int inner_row_b,int inner_col_b){
    //搬运过程
    //不对啊  v4版本是 一个线程根据 寄存器的容量 去决定搬运几个float4,那这个到底搬运几个？难道是根据reg_m   reg_n的大小？
    //从代码中看 这个是一个搬运一个float4，不像v4版本
    //我有点恍然大明白了，v4需要一个临时寄存器 是因为一次搬运多个！ 而这里不需要搬运多个  那么只需要tmp得了！
    //开始搬运吧 ！  一共thread_nums个线程 ，如何定位呢？这里不像V4了，思想应该像是V3了。
    //如何定位呢？ inner_row_a 已经给出来了
    for(uint offset =0 ; offset+row_stride_a <=BM ; offset +=row_stride_a){
        //offset+row_stride_a是真正的行。
        //inner_col_a线程定位  所以要乘以4才是数据的真实位置。
        const float4 tmp = reinterpret_cast<const float4*>(&A[(inner_row_a+offset)*K+inner_col_a*4])[0];

        //现在是offset+row_stride_a行， inner_col_a列
        //要转置存储
        As[(inner_col_a +0)*BM+inner_col_a+offset] = tmp.x;
        As[(inner_col_a +1)*BM+inner_col_a+offset] = tmp.y;
        As[(inner_col_a +2)*BM+inner_col_a+offset] = tmp.z;
        As[(inner_col_a +3)*BM+inner_col_a+offset] = tmp.w;
    }
    for(uint offset=0; offset+row_stride_b <=BN ; offset +=row_stride_b){
//        const float4 Bs[(inner_row_b+offset)*BN+inner_col_b*4]] = reinterpret_cast<const float4*>(&B[(inner_row_b+offset)*N+inner_col_b*4])[0];
        reinterpret_cast< float4*> (&Bs[(inner_row_b+offset )*BN+inner_col_b*4])[0] = reinterpret_cast<const float4*>(&B[(inner_row_b+offset)*N+inner_col_b*4])[0];
    }

}

template<const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER,const int WNITER,const int WSUBM,const int WSUBN,const int TM,const int TN>
__device__ void process_from_smem(float *reg_m, float* reg_n, float* thread_results,const float *As, const float*Bs,const uint warp_row, const uint warp_col,const uint thread_row_in_warp,const uint thread_col_inwarp)
{
    //现在已经搬运完了A B 子块到As Bs里边了 开始算
    for(uint dot_idx =0 ; dot_idx< BK; dot_idx++){
        //首先要准备计算所需的数据 就是图中的小蓝色条
        //搬运As的小蓝条
        for(uint w_sub_row_idx =0 ; w_sub_row_idx< WMITER; w_sub_row_idx++){ //w_sub_row_idx是指搬运第几个小蓝条
            for(uint i =0 ; i<TM; i++){//这里为啥不用float4向量化读取呢？
                //无非是对应行 列罢了  下边这个第一项事第几行，后边是关于列的定位  这是自己写的
//                reg_m[w_sub_row_idx*TM+i] =As[dot_idx*BM + thread_row_in_warp*TM + w_sub_row_idx*WSUBM + i];
                //正确答案
                reg_m[w_sub_row_idx * TM + i] =As[(dot_idx * BM) + warp_row * WM + w_sub_row_idx * WSUBM +thread_row_in_warp * TM + i];
            }
        }

        //搬运Bs的小蓝条
        for(uint w_sub_col_idx =0 ; w_sub_col_idx< WNITER; w_sub_col_idx++){
            for(uint i =0 ; i<TN; i++){
                //同样 无非是行与列的定位罢了
                reg_n[w_sub_col_idx*TN +i] = Bs[dot_idx*BN + warp_col*WN + w_sub_col_idx*WSUBN + thread_col_inwarp*TN + i];
            }
        }
        //开始计算 WMITER * WNITER *TM*TN的结果块，
        for (uint w_sub_row_idx =0 ; w_sub_row_idx< WMITER; w_sub_row_idx++){
            for (uint w_sub_col_idx =0 ; w_sub_col_idx< WNITER; w_sub_col_idx++){
                for(uint res_idx_m=0;res_idx_m < TM;res_idx_m++){
                    for(uint res_idx_n=0;res_idx_n < TN;res_idx_n++){
                        //首先要定位到左上角顶点，然后才是 第res_idx_m行，第res_idx_n列的事儿，不对它这个要考虑第几行的，所以应该考虑的是小黄豆 的定位
                        thread_results[w_sub_row_idx*WNITER*TM*TN+res_idx_m*WNITER*TN+w_sub_col_idx*TN+res_idx_n] += reg_m[w_sub_row_idx*TM+res_idx_m] * reg_n[w_sub_col_idx*TN+res_idx_n];
                    }
                }
                
            }
        }

    }
}

constexpr int WARP_SIZE =32;
template<const int BM,const int BN, const int BK, const int WM,const int WN,  const int WNITER, const int TM, const int TN,const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
mysgemm_warptiling(int M, int N, int K,float aplha, float *A,float *B,float beta,float*C){
    const uint c_row = blockIdx.x; //一个线程块负责一个C的子块
    const uint c_col = blockIdx.y;

    const uint warp_idx = threadIdx.x / WARP_SIZE; //感觉warp是一个逻辑上的编号
    const uint warp_col = warp_idx % (BN/WN); //warp的二维索引
    const uint warp_row = warp_idx / (BN/WN);

    constexpr uint WMITER = (WM*WN)/(WARP_SIZE*TM*TN*WNITER); //这个显而易见么 这些值都是设定的
    constexpr int WSUBM = WM/WMITER;
    constexpr int WSUBN = WN/WMITER;

    const uint thread_idx_in_warp = threadIdx.x % WARP_SIZE;  //线程束编号
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN); //该线程在线程束中的定位
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    __shared__ float As[BM*BK];
    __shared__ float Bs[BN*BK];

    //挪到该块处理的位置
    A += c_row *BM *K;  //注意此处写错 自己写成A = c_row *BM*K； 这给整成整数了 ，应该是挪动指针,可以通过A= &A[]这种写法
    B += c_col*BN;
//    C += c_row*BM*N+ c_col*BN;     //此处一遍一遍的忘  该处写错了
    C +=  c_row *BM * N + c_col*BN + warp_row*WN*BN + warp_col*WN;

    //开始搬运 A子块  B子块到共享显存，此处应该跟V4是一样的， 此处可以看第7页PPT
    const uint inner_row_a = threadIdx.x /(BK /4); // (BK/4) 是指一行需要多少个线程读取（因为float4读取，一个线程读4个数据）
    const uint inner_col_a = threadIdx.x %(BK /4); //线程第一次读取float4 位于二维的第几行
    constexpr uint row_stride_a = (NUM_THREADS*4)/BK; //就是一个线程搬运数据的时候下一个步长

    const uint inner_row_b = threadIdx.x /(BN /4);
    const uint inner_col_b = threadIdx.x %(BN /4);
    constexpr uint row_stride_b = NUM_THREADS /(BN /4);

    float thread_results[WMITER*TM*WNITER*TN] = {0.0}; //此处是每个线程负责计算的C子块的累加结果
    float reg_m[WMITER*TM] = {0.0}; //每个线程计算的时候用到的寄存器
    float reg_n[WNITER*TN] = {0.0}; //每个线程计算的时候用到的寄存器

    //开始遍历，改变里是搬运一块 计算一块
    //这里不明白 处理的结果放哪里去了？ 而且是放转置矩阵的时候该有寄存器啊我记得
    for(uint bk_idx =0; bk_idx < K; bk_idx += BK){
        //搬运A子块到共享显存
        load_from_gmem<BN,BM,BK,row_stride_a,row_stride_b>(N,K,A,B,As,Bs,inner_row_a,inner_col_a,inner_row_b,inner_col_b);
        __syncthreads(); //这里搬运完了 需要等待所有线程都运行到这里才能开始计算。

        //开始计算
        process_from_smem<BM,BN,BK,WM,WN,WMITER,WNITER,WSUBM,WSUBN,TM,TN>(reg_m,reg_n,thread_results,As,Bs,thread_row_in_warp,thread_col_in_warp,thread_idx_in_warp,warp_row,warp_col);
        A+=BK;
        B+=BK*N;
        __syncthreads(); //计算完了，需要等待所有线程都运行到这里才能开始搬运下一块

        //现在计算完了 得把thread_results中的结果写回C矩阵啊 要写回也简单啊，不就是把这几个TM*TN的块写回去吗
        for(uint w_sub_row_idx =0; w_sub_row_idx < WMITER; w_sub_row_idx++){
            for(uint w_sub_col_idx =0; w_sub_col_idx < WNITER; w_sub_col_idx++){
                //这个其实还是一个定位问题  定位TM*TN的小蓝块  要考虑什么呢？
                //要考虑哪个warp 哪个迭代  线程在warp中的位置
                //已知C已经指向 该warp负责的范围起点了。  那就好办了

                //w_sub_col_idx  w_sub_row_idx 既代表着该warp负责的第几个分片，又是结果中存的 WMITER*WNITER*TM*TN的索引
                //C_interim 指向：该 warp tile 中第 (w_sub_row_idx, w_sub_col_idx) 个小分片（WSUBM × WSUBN）的左上角。
                float *C_interim = C + (w_sub_row_idx*WSUBM)*N  + w_sub_col_idx*WSUBN; //这个是定位到该warp负责的子块的第几个分片

                //下边两个循环是用来访问TM*TN的，用 res_idx_m / res_idx_n 访问每 4 元素
                for(uint res_idx_m =0; res_idx_m < TM; res_idx_m++){
                    for(uint res_idx_n =0; res_idx_n < TN; res_idx_n+=4){ //这里为什么+4呢？
                        //先拿这个结果对应的 C的位置  ， 一下拿四个元素
//                        float4 tmp =reinterpret_cast<float *>(&C_interim[(thread_row_in_warp+res_idx_m)*N+thread_col_in_warp*TN+res_idx_n])[0];
                        float4 tmp = reinterpret_cast<float4*>(&C_interim[(thread_row_in_warp *TM+res_idx_m)*N + thread_col_in_warp*TN +res_idx_n])[0];

                        //拿计算结果
                        //定义一个索引，要拿thread_results中的结果了。这里就相当于PPT中黄色快thread_results中的TM*TN如何定位
                        //已知 w_sub_row_idx  w_sub_col_idx  res_ix_m  res_idx_n
                        const int i = (w_sub_row_idx*TM*WNITER+res_idx_m)*TN + w_sub_col_idx*TN + res_idx_n;

                        //把结果写到tmp里边去
                        tmp.x = aplha*thread_results[i+0] + beta*tmp.x;
                        tmp.y = aplha*thread_results[i+1] + beta*tmp.y;
                        tmp.z = aplha*thread_results[i+2] + beta*tmp.z;
                        tmp.w = aplha*thread_results[i+3] + beta*tmp.w;
                        //写回去
                        reinterpret_cast<float4 *>(&C_interim[(thread_row_in_warp * TM + res_idx_m) * N +thread_col_in_warp * TN + res_idx_n])[0] = tmp;

                    }
                }



            }
        }
    }
}

std::vector<int> generateSizes(){
    std::vector<int> sizes;
    for (int i=256; i<= 8192; i+=256){
        sizes.push_back(i);
    }
    return sizes;
    }
#define CEIL_DIV(M,N) ((M)+(N)-1)/(N)

int main(){
    std::vector<int> sizes = generateSizes();
    //打开csv文件
    std::ofstream csv_file("/cuda_code/tmp/sgemm_benchmark_v7.csv");
    csv_file << "Size ,CUBLAS_GFLOPS, MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    for (int N : sizes){
        std::cout << "Size: " << N << std::endl;

        size_t size =N *N *sizeof(float);

        float *A = (float *)malloc(size);
        float *B = (float *)malloc(size);
        float *C = (float *)malloc(size);
        float *C_ref = (float *)malloc(size);

        float *d_A , *d_B , *d_C_v1;
        checkCudaError(cudaMalloc(&d_A,size),"cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B,size),"cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1,size),"cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;
        try{
            for(int i=0;i<N*N ;i++){
                A[i] = 1.0f;
                B[i] = 2.0f;
            }
            checkCudaError(cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice),"cudaMemcpy d_A failed");
            checkCudaError(cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice),"cudaMemcpy d_B failed");

            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            cudaEvent_t start,stop;
            checkCudaError(cudaEventCreate(&start),"cudaEventCreate start failed");
            checkCudaError(cudaEventCreate(&stop),"cudaEventCreate stop failed");

            int warpup_time =10;
            for (int i = 0; i < warpup_time; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                             &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();




        }

    }
}