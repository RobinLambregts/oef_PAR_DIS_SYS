#include "image2d.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <iostream>

#define BLOCKSIZE 16

__global__ void CUDAKernel(int iterations, float xmin, float xmax, float ymin, float ymax, 
                           float *pOutput, int outputW, int outputH)
{
	// TODO: your code
}

// If an error occurs, return false and set a description in 'errStr'
bool cudaFractal(int iterations, float xmin, float xmax, float ymin, float ymax, 
                 Image2D &output, std::string &errStr)
{
	// We'll use an image of 512 pixels wide
	int ho = 512;
	int wo = ho * 3 / 2;
	output.resize(wo, ho);

	// And divide this in a number of blocks
	size_t xBlockSize = BLOCKSIZE;
	size_t yBlockSize = BLOCKSIZE;
	size_t numXBlocks = (wo/xBlockSize) + (((wo%xBlockSize) != 0)?1:0);
	size_t numYBlocks = (ho/yBlockSize) + (((ho%yBlockSize) != 0)?1:0);

	cudaError_t err;
	float *pDevOutput;

	// TODO: allocate memory on GPU

	cudaEvent_t startEvt, stopEvt; // We'll use cuda events to time everything
	cudaEventCreate(&startEvt);
	cudaEventCreate(&stopEvt);

	cudaEventRecord(startEvt);

	// TODO: call kernel

	cudaEventRecord(stopEvt);
	
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "CUDA convolution kernel execution error code: " << err << std::endl;
	}

	// TODO: copy data back and free memory
	//       Note that output.getBufferPointer() gives direct access to the
	//       floating point array in the image

	float elapsed;
	cudaEventElapsedTime(&elapsed, startEvt, stopEvt);

	std::cout << "CUDA time elapsed: " << elapsed << " milliseconds" << std::endl;

	cudaEventDestroy(startEvt);
	cudaEventDestroy(stopEvt);

	return true;
}
