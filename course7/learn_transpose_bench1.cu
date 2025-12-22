#include <iostream>
#include <cuda_runtime.h>
//该代码是 使用共享显存  bank冲突的代码


#define BDIMX 32
#define BDIMY 16


__global__ void transposeSmem(float *out, float *in, const int nx, const int ny)
{

   __shared__ float tile[BDIMY][BDIMX];
   unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;
   unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;

   //
   unsigned int ti = iy*nx + ix;
   //查看块内索引
   unsigned int bidx = threadIdx.y*blockDim.x + threadIdx.x;
   unsigned int irow = bidx / blockDim.y;
   unsigned int icol = bidx % blockDim.y;

   //在转置矩阵内的索引
   ix = blockIdx.y * blockDim.y + icol;
   iy = blockIdx.x * blockDim.x + irow;
   unsigned int to = iy*ny + ix;
   if(ix<nx && iy<ny){
       tile[threadIdx.y][threadIdx.x] = in[ti];
       __syncthreads();
       out[to] = tile[icol][irow];
   }
}


__global__ void transposeSmemUnrollPad(float*out ,float*in,int nx, int ny){
    const int IPAD = 1;
    __shared__ float tile[BDIMY*(BDIMX*2+IPAD)];
    unsigned int ix = (2*blockDim.x)*blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;  //
    unsigned int ti = iy*nx + ix;

    unsigned int bidx = threadIdx.y *blockDim.x+threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    unsigned int ix2 = blockIdx.y*blockDim.y + icol;
    unsigned int iy2 = 2*blockIdx.x*blockDim.x + irow; //这个二也不是很理解啊 上个2是不重叠，这个2 按照转置来写吧，在图上不好表示啊
    unsigned int to = iy2*ny + ix2;
    if((ix+blockDim.x)<nx && iy<ny){ //这个条件是啥意思啊
        unsigned int row_idx = threadIdx.y*(blockDim.x*2+IPAD) + threadIdx.x; //上个是二维的 这个 算子用的一维的
        tile[row_idx] = in[ti];
        tile[row_idx+BDIMX] = in[ti+BDIMX];
        __syncthreads();
        unsigned int col_idx = icol*(blockDim.x*2+IPAD) + irow;
        out[to] = tile[col_idx];
        out[to+ny*BDIMX] = tile[col_idx+BDIMX];
        }

}



void call_transposeSmem(float *d_out,float *d_in,int nx,int ny) {
    dim3 blockSize(BDIMX,BDIMY);
    dim3 gridSize((nx + BDIMX - 1) / BDIMX,(ny + BDIMY - 1) / BDIMY);

    // Launch the kernel
    transposeSmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);

}



void naiveSmemWrapper()
{
    int nx = 4096;
    int ny = 4096;
    size_t size = nx* ny * sizeof(float);

    //主机内存分配
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    //初始化输入矩阵
    for(int i=0;i< nx*ny;i++){
        h_in[i] = float(int(i)%11);
    }
    //设备内存分配
    float *d_in,*d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out,size);

    //将数据从主机复制到设备
    cudaMemcpy(d_in,h_in,size,cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warp_up_iter = 5;
    for(int i=0;i<warp_up_iter;++i){
        call_transposeSmem(d_out,d_in,nx,ny);
    }

    int bench_iter =5;

    //开始计时
    cudaEventRecord(start);
    for(int i=0;i<bench_iter;++i){
        call_transposeSmem(d_out,d_in,nx,ny);
    }
    //结束计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess){
        std::cerr << "CUDA ERROER: " << cudaGetErrorString(err) << std::endl;
        return ;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    std::cout << "Smem transpose kernel execution time:" << milliseconds << " ms" << std::endl;

    //将结果从设备复制到主机
    cudaMemcpy(h_out,d_out,size ,cudaMemcpyDeviceToHost);
    //释放内存
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Matrix transposition completed successfully." << std::endl;
    }




int main()
{
    naiveSmemWrapper();
}