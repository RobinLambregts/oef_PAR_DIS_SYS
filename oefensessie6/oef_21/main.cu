#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeIndices(int *blockArray, int *threadArray, int length) {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < length) {
        blockArray[globalIndex] = blockIdx.x;
        threadArray[globalIndex] = threadIdx.x;
    }
}

int main() {
    const int length = 500;
    const int blocks = 16;
    const int threadsPerBlock = (length + blocks - 1) / blocks;

    int *blockResult = new int[length];
    int *threadResult = new int[length];

    int *d_blockResult, *d_threadResult;
    cudaMalloc(&d_blockResult, length * sizeof(int));
    cudaMalloc(&d_threadResult, length * sizeof(int));

    writeIndices<<<blocks, threadsPerBlock>>>(d_blockResult, d_threadResult, length);
    cudaDeviceSynchronize();

    cudaMemcpy(blockResult, d_blockResult, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(threadResult, d_threadResult, length * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Index\tBlock\tThread\n");
    for (int i = 0; i < length; i++) {
        printf("%3d\t%3d\t%3d\n", i, blockResult[i], threadResult[i]);
    }

    delete[] blockResult;
    delete[] threadResult;
    cudaFree(d_blockResult);
    cudaFree(d_threadResult);

    return 0;
}
