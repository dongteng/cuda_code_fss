#include <stdio.h>
#include <cuda_runtime.h>

__global__ void race_condition_kernel(int *data){
    int temp = *data;
    temp += 1;
    *data = temp;

}

int main(){
    int *d_data ;
    int h_data = 0;

    cudaMalloc(&d_data, sizeof(int));

    cudaMemcpy(d_data,&h_data, sizeof(int),cudaMemcpyHostToDevice);

    //使用1024线程块，每个块256个线程
    race_condition_kernel<<<1024,256>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data,d_data,sizeof(int),cudaMemcpyDeviceToHost);

    printf("h_data = %d\n",h_data);
    cudaFree(d_data);
    return 0;
}