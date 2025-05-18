#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "flipper.h"    //include this header with a proper path
#include "stdio.h"
#include <iostream>

using namespace std;

__global__ void reversingKernel(unsigned char* data, int flipSize)
{
    unsigned long long  threadStartPoint = blockIdx.x * flipSize + threadIdx.x;                      // Left side index
    unsigned long long  threadRespectPoint = blockIdx.x * flipSize + (flipSize - threadIdx.x - 1);   // Rigth side index
    unsigned char tmp = data[threadStartPoint];
    data[threadStartPoint] = data[threadRespectPoint];
    data[threadRespectPoint] = tmp;
}

/*!
\param inputData      Data which is going to be flipped , the output will be also saved in input data
\param inputDataSize  Size of input data
\param flipSize       The size of flipping
\param outputsize     The size of output
*/
unsigned char* flipper(unsigned char* inputData, long long inputDataSize, int flipSize, long long& outputsize)
{
    unsigned char* reshapedData;
    unsigned char* inputData_device;
    unsigned char* outData;

    dim3 threadsDim(flipSize / 2, 1, 1);                                    // Dimension of threads
    dim3 BlocksDim((int)(inputDataSize / ((long long)flipSize)), 1, 1);     // Dimension of Blocks

    if (inputDataSize % flipSize != 0)                                      // Checks to see inputDataSize can be counted by flipSize or not
    {
        long long newSize = inputDataSize - (inputDataSize % flipSize);    // New size of data
        reshapedData = new unsigned char[newSize];
        outData = new unsigned char[newSize];
        outputsize = newSize;

        for (long long i = 0; i < newSize; ++i)
            reshapedData[i] = inputData[i];

        // Allocating buffer on GPU
        cudaMalloc((void**)&inputData_device, newSize * sizeof(unsigned char));

        // Copy Data from CPU to GPU
        cudaMemcpy(inputData_device, reshapedData, newSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Kernel lunch
        reversingKernel <<<BlocksDim, threadsDim >>> (inputData_device, flipSize);

        // Copy Data from GPU to CPU
        cudaMemcpy(outData, inputData_device, newSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Free allocated memories
        cudaFree(inputData_device);
        delete[] reshapedData;
    }

    else
    {
        outData = new unsigned char[inputDataSize];
        outputsize = inputDataSize;

        // Allocating buffer on GPU
        cudaMalloc((void**)&inputData_device, inputDataSize * sizeof(unsigned char));

        // Copy Data from CPU to GPU
        cudaMemcpy(inputData_device, inputData, inputDataSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Kernel lunch
        reversingKernel <<<BlocksDim, threadsDim >>> (inputData_device, flipSize);

        // Copy Data from GPU to CPU
        cudaMemcpy(outData, inputData_device, inputDataSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Free allocated memories
        cudaFree(inputData_device);
    }

    // Output
    return outData;
}
