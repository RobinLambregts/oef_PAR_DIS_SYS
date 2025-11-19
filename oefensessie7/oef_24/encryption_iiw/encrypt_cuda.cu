#include <vector>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <stdio.h>
#include "parameters.h"

inline void cudaCheckError(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(code));
        exit(code);
    }
}

// GPU F-functie
__device__ void runF(uint8_t* output, const uint8_t* input, const uint8_t* key) {
    uint8_t tmp[512];

    // XOR input met key
    for (int i = 0; i < 512; ++i)
        tmp[i] = input[i] ^ key[i];

    // Substitutie
    for (int i = 0; i < 512; ++i)
        tmp[i] = substitutionTable_cuda[tmp[i]];

    // Permutatie
    for (int i = 0; i < 512; ++i)
        output[permutationTable_cuda[i]] = tmp[i];
}

__global__ void feistelKernel(uint8_t* data,
                              const uint8_t* roundkeys,
                              int numRounds,
                              bool decrypt,
                              int dataSize)
{
    int chunkIdx = blockIdx.x;
    int base = chunkIdx * 1024;

    if (base >= dataSize)
        return;

    int chunkSize = min(1024, dataSize - base);

    // L is altijd <= 512 bytes
    int Lsize = min(512, chunkSize);

    // R bevat enkel bytes indien chunkSize > 512
    int Rsize = (chunkSize > 512 ? chunkSize - 512 : 0);

    __shared__ uint8_t L[512];
    __shared__ uint8_t R[512];
    __shared__ uint8_t Fout[512];

    int tid = threadIdx.x;

    // laad L
    if (tid < Lsize)
        L[tid] = data[base + tid];

    // laad R
    if (tid < Rsize)
        R[tid] = data[base + 512 + tid];

    __syncthreads();

    int startKey = decrypt ? numRounds - 1 : 0;
    int stopKey  = decrypt ? -1 : numRounds;
    int step     = decrypt ? -1 : 1;

    for (int k = startKey; k != stopKey; k += step) {

        if (tid == 0) {
            // F gebruikt altijd volledige 512-bytes blok, maar leest alleen echte R bytes
            uint8_t paddedR[512];

            // kopieer echte R
            for (int i = 0; i < Rsize; i++)
                paddedR[i] = R[i];

            // pad rest met nullen
            for (int i = Rsize; i < 512; i++)
                paddedR[i] = 0;

            runF(Fout, paddedR, &roundkeys[k * 512]);
        }
        __syncthreads();

        // XOR alleen echte L-bytes
        if (tid < Lsize)
            L[tid] ^= Fout[tid];

        __syncthreads();

        bool lastRound =
            (!decrypt && k == numRounds - 1) ||
            (decrypt && k == 0);

        // swap behalve laatste ronde
        if (!lastRound) {
            if (tid < min(Lsize, Rsize)) {
                uint8_t t = L[tid];
                L[tid] = R[tid];
                R[tid] = t;
            }
        }
        __syncthreads();
    }

    // Schrijf terug exact aantal bytes
    if (tid < Lsize)
        data[base + tid] = L[tid];

    if (tid < Rsize)
        data[base + 512 + tid] = R[tid];
}

std::vector<uint8_t> encrypt_cuda(std::vector<uint8_t> plaintext,
                                  std::vector<std::vector<uint8_t>> roundkeys,
                                  bool decrypt)
{
    int dataSize = (int)plaintext.size();
    std::vector<uint8_t> output(dataSize);

    int numChunks = (dataSize + 1024 - 1) / 1024;
    int numRounds = roundkeys.size();

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

    cudaCheckError(cudaMemcpy(output.data(), d_data, dataSize, cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_roundkeys);

    return output;
}
