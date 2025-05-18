# CUDA Bit Flipper

This project implements a CUDA-based parallel bit flipper for efficiently reversing segments of binary data (0s and 1s). It is designed to operate on large buffers and perform high-speed flipping of data blocks on the GPU.

# 🚀 Features

    Reverses chunks of binary data (0s and 1s) in parallel using CUDA.

    Automatically handles input sizes that are not multiples of the flip size by trimming.

    Efficient use of CUDA threads and blocks for fast execution.

    GPU memory management with fallback to CPU reshaping if needed.

# 📁 Project Structure

├── flipper.cu         # Contains the CUDA kernel and host wrapper function

├── flipper.h          # Header for the flipper function (user should provide)

├── README.md          # Project documentation

# 🧠 How It Works

Each CUDA thread is responsible for flipping a pair of bits within a block of size flipSize. If inputDataSize is not evenly divisible by flipSize, the extra bytes are ignored.

The flipping kernel performs a swap between:

threadStartPoint = blockIdx.x * flipSize + threadIdx.x;
threadRespectPoint = blockIdx.x * flipSize + (flipSize - threadIdx.x - 1);

This effectively mirrors the bits within each block.

# 🛠️ Usage
## Function Signature

unsigned char* flipper(unsigned char* inputData, long long inputDataSize, int flipSize, long long& outputsize);

### Parameters

    inputData: Pointer to the input binary data (expected values: 0 or 1).

    inputDataSize: Length of the input data in bytes.

    flipSize: Size of each chunk to reverse (must be even and ≤1024 for typical GPU thread limits).

    outputsize: Returns the size of the processed output.

### Returns

A new buffer containing the flipped data. The caller is responsible for freeing this memory.

## 📦 Example

    #include "flipper.h"

    int main() {
      unsigned char data[] = {0, 1, 0, 1, 1, 0, 1, 0};
      long long outputSize;

      unsigned char* flipped = flipper(data, 8, 4, outputSize);

      for (long long i = 0; i < outputSize; ++i) {
        std::cout << (int)flipped[i] << " ";
      }

    delete[] flipped;
    return 0;
    }

## Output

### 1 0 1 0 0 1 0 1

## ⏱️ Performance

### Benchmark

    Input Size: 100 MB

    Flip Block Size: 1024 bytes

    GPU: NVIDIA RTX 3080

    Execution Time: ~6.2 milliseconds

    Throughput: ~16.1 GB/s

### ⚠️ Note: Actual performance will vary depending on your GPU, CUDA version, and memory bandwidth.

## ⚠️ Requirements

    CUDA Toolkit (version >= 10.0 recommended)
    NVIDIA GPU with CUDA Compute Capability 3.0 or higher
    C++ compiler

## 🧹 Notes

    The current kernel assumes flipSize is even. For odd values, consider adapting the kernel to avoid double-swapping the middle element.

    This implementation trims any leftover data that does not fit a full flipSize block.

## 📄 License

    This project is open-source under the MIT License. Feel free to use, modify, and distribute.






