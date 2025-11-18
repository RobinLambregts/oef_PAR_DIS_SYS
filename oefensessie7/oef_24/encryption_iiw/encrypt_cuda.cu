
#include <vector>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "parameters.h"

inline void cudaCheckError(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s
", cudaGetErrorString(code));
        exit(code);
    }
}

__device__ void runF(uint8_t* output, const uint8_t* input, const uint8_t* key) {
    uint8_t tmp[512];

    for (int i = 0; i < 512; ++i)
        tmp[i] = input[i] ^ key[i];

    for (int i = 0; i < 512; ++i)
        tmp[i] = substitutionTable_cuda[tmp[i]];

    for (int i = 0; i < 512; ++i) {
        int j = permutationTable_cuda[i];
        output[j] = tmp[i];
    }
}

__global__ void feistelKernel(uint8_t* data,
                              const uint8_t* roundkeys,
                              int numRounds,
                              bool decrypt,
                              int dataSize)
{
    int chunkIdx = blockIdx.x;
    if (chunkIdx * 1024 >= dataSize) return;

    __shared__ uint8_t L[512];
    __shared__ uint8_t R[512];
    __shared__ uint8_t Fout[512];

    int base = chunkIdx * 1024;
    int tid = threadIdx.x;

    if (tid < 512) {
        L[tid] = data[base + tid];
        R[tid] = data[base + 512 + tid];
    }
    __syncthreads();

    int startKey = decrypt ? numRounds - 1 : 0;
    int stopKey  = decrypt ? -1 : numRounds;
    int step     = decrypt ? -1 : 1;

    for (int k = startKey; k != stopKey; k += step) {
        if (tid == 0) {
            runF(Fout, R, &roundkeys[k * 512]);
        }
        __syncthreads();

        if (tid < 512) {
            L[tid] ^= Fout[tid];
        }
        __syncthreads();

        bool lastRound = (!decrypt && k == numRounds - 1) || (decrypt && k == 0);
        if (!lastRound) {
            if (tid < 512) {
                uint8_t temp = L[tid];
                L[tid] = R[tid];
                R[tid] = temp;
            }
        }
        __syncthreads();
    }

    if (tid < 512) {
        data[base + tid] = L[tid];
        data[base + 512 + tid] = R[tid];
    }
}

std::vector<uint8_t> encrypt_cuda(std::vector<uint8_t> plaintext,
                                  std::vector<std::vector<uint8_t>> roundkeys,
                                  bool decrypt = false)
{
    int dataSize = (int)plaintext.size();
    if (dataSize % 1024 != 0)
        throw std::runtime_error("plaintext size must be multiple of 1024");

    int numChunks = dataSize / 1024;
    int numRounds = (int)roundkeys.size();

    std::vector<uint8_t> flatKeys(numRounds * 512);
    for (int i = 0; i < numRounds; ++i)
        memcpy(&flatKeys[i * 512], roundkeys[i].data(), 512);

    uint8_t* d_data = nullptr;
    uint8_t* d_roundkeys = nullptr;

    cudaCheckError(cudaMalloc(&d_data, dataSize));
    cudaCheckError(cudaMemcpy(d_data, plaintext.data(), dataSize, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc(&d_roundkeys, flatKeys.size()));
    cudaCheckError(cudaMemcpy(d_roundkeys, flatKeys.data(), flatKeys.size(), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(512);
    dim3 numBlocks(numChunks);

    feistelKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_roundkeys, numRounds, decrypt, dataSize);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    std::vector<uint8_t> output(dataSize);
    cudaCheckError(cudaMemcpy(output.data(), d_data, dataSize, cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_roundkeys);

    return output;
}
