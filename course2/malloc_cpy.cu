#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    int size = 4 * sizeof(int);
    int *h_a = (int *) malloc(size);
    int *h_b = (int *) malloc(size);
    int *d_a;

    for (int i = 0; i < 4; i++) {
        h_a[i] = i;
    }
    cudaMalloc((void **) &d_a, size);//cudaMalloc就这么规定的！ 第一个参数就是指向 void指针的 指针
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i) {
        printf("h_b[%d] = %d\n", i, h_b[i]);
    }

    cudaFree(d_a);
    free(h_a);
    free(h_b);
    return 0;
}
