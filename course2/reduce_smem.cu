#include <iostream>

//block内并行规约求和kernel
//g_idata: 输入数组（放在 global memory）；g_odata: 输出数组，每个 block 会写一个值进去；n: 输入数组的元素个数
__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x; //threadIdx.x的类型其实原本就是unsigned int类型
    if (tid >= n)
        return;

    //每个 block 处理 blockDim.x 个元素；
    //所以 block 0 处理 [0, 1, ..., blockDim.x-1]  block 1 处理 [blockDim.x, ..., 2*blockDim.x-1]
    //idata 就是当前 block 自己的数据段首地址。
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //并行规约阶段 1   表示这个线程块中有 至少 1024 个线程  且 只有前512个线程会执行 idata[tid] += idata[tid + 512]；
    //tid = 0       idata[0] += idata[512]
    //tid = 1       idata[1] += idata[513]
    //...
    //tid = 511     idata[511] += idata[1023]
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();   //所有线程都在这里同步等待，直到所有线程都走到这一行
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
    //为什么该处写的跟上边不一样？因为最后边32个线程同属于同一个warp
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n) {
    __shared__ int smem[256];
    unsigned int tid = threadIdx.x;
    if (tid >= n)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    smem[tid] = idata[tid];
    __syncthreads();
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

int main() {
    const int n = 102400;
    int h_idata[n];

    for (int i = 0; i < n; i++)
        h_idata[i] = 1;

    int *d_odata1, *d_odata2, *d_idata1, *d_idata2;
    cudaMalloc((void **) &d_idata1, n * sizeof(int));
    cudaMalloc((void **) &d_idata2, n * sizeof(int));
    cudaMalloc((void **) &d_odata1, sizeof(int));
    cudaMalloc((void **) &d_odata2, sizeof(int));

    cudaMemcpy(d_idata1, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idata2, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Warmup kernels
    for (int i = 0; i < 5; ++i) {
        reduceGmem<<<numBlocks, blockSize>>>(d_idata1, d_odata1, n);
        reduceSmem<<<numBlocks, blockSize>>>(d_idata2, d_odata2, n);
    }
    //重新把原始数据拷贝到 GPU（防止预热修改了输入）
    cudaMemcpy(d_idata1, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idata2, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);

    // Event creation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure reduceGmem
    cudaEventRecord(start);
    reduceGmem<<<numBlocks, blockSize>>>(d_idata1, d_odata1, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); //堵塞CPU， 等待GPU执行完stop时间之后 (kernel结束)
    float millisecondsGmem = 0;
    cudaEventElapsedTime(&millisecondsGmem, start, stop);
    std::cout << "Time for reduceGmem: " << millisecondsGmem << " ms"
              << std::endl;

    // Measure reduceSmem
    cudaEventRecord(start);
    reduceSmem<<<numBlocks, blockSize>>>(d_idata2, d_odata2, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float millisecondsSmem = 0;
    cudaEventElapsedTime(&millisecondsSmem, start, stop);
    std::cout << "Time for reduceSmem: " << millisecondsSmem << " ms"
              << std::endl;

    int h_odata1, h_odata2;
    cudaMemcpy(&h_odata1, d_odata1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_odata2, d_odata2, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum1: " << h_odata1 << std::endl;
    std::cout << "Sum2: " << h_odata2 << std::endl;

    // Cleanup
    cudaFree(d_idata1);
    cudaFree(d_idata2);
    cudaFree(d_odata1);
    cudaFree(d_odata2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
