#include "image2d.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <iostream>

#define BLOCKSIZE 16

__global__ void CUDAKernelFractal(int iterations, float xmin, float xmax, float ymin, float ymax,
                                  float *pOutput, int outputW, int outputH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outputW || y >= outputH) return;

    float dx = (xmax - xmin) / outputW;
    float dy = (ymax - ymin) / outputH;

    float cx = xmin + x * dx;
    float cy = ymin + y * dy;

    float zx = 0.0f;
    float zy = 0.0f;

    int n = 0;
    while (n < iterations && (zx*zx + zy*zy <= 4.0f)) {
        float xt = zx*zx - zy*zy + cx;
        zy = 2.0f*zx*zy + cy;
        zx = xt;
        n++;
    }

    float value;
    if (n == iterations) {
        value = 0.0f; // zwart
    } else {
        value = sqrt(n / (float)iterations) * 255.0f;
    }

    pOutput[y * outputW + x] = value;
}

bool cudaFractal(int iterations, float xmin, float xmax, float ymin, float ymax,
                 Image2D &output, std::string &errStr)
{
    int ho = 512;
    int wo = ho * 3 / 2;
    output.resize(wo, ho);

    size_t numXBlocks = (wo + BLOCKSIZE - 1) / BLOCKSIZE;
    size_t numYBlocks = (ho + BLOCKSIZE - 1) / BLOCKSIZE;

    float *pDevOutput;
    cudaError_t err = cudaMalloc((void**)&pDevOutput, wo * ho * sizeof(float));
    if (err != cudaSuccess) {
        errStr = "CUDA malloc failed";
        return false;
    }

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid(numXBlocks, numYBlocks);

    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);
    cudaEventRecord(startEvt);

    CUDAKernelFractal<<<grid, block>>>(iterations, xmin, xmax, ymin, ymax, pDevOutput, wo, ho);

    cudaEventRecord(stopEvt);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        errStr = "CUDA kernel execution failed";
        cudaFree(pDevOutput);
        return false;
    }

    cudaMemcpy(output.getBufferPointer(), pDevOutput, wo * ho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pDevOutput);

    float elapsed;
    cudaEventElapsedTime(&elapsed, startEvt, stopEvt);
    std::cout << "CUDA time elapsed: " << elapsed << " milliseconds" << std::endl;

    cudaEventDestroy(startEvt);
    cudaEventDestroy(stopEvt);

    return true;
}
