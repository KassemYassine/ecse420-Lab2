#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gputimer.h"
#include "lodepng.h"
#include "wm.h"

#define CLAMP_0_255(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x)))

__constant__ float d_w[3][3];

__global__
void gpuConvolveKernel(const unsigned char* d_in,
                       unsigned char*       d_out,
                       unsigned width,
                       unsigned height)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInner = (width - 2) * (height - 2);
    if(gid >= totalInner) return;
    int outRow = gid / (width - 2);
    int outCol = gid % (width - 2);
    int inRow = outRow + 1;
    int inCol = outCol + 1;
    float rVal = 0.f;
    float gVal = 0.f;
    float bVal = 0.f;
    for(int ii = 0; ii < 3; ii++)
    {
        for(int jj = 0; jj < 3; jj++)
        {
            int rr = inRow + (ii - 1);
            int cc = inCol + (jj - 1);
            int inIdx = (rr * width + cc) * 4;
            rVal += d_in[inIdx + 0] * d_w[ii][jj];
            gVal += d_in[inIdx + 1] * d_w[ii][jj];
            bVal += d_in[inIdx + 2] * d_w[ii][jj];
        }
    }
    int outIdx = (outRow * (width - 2) + outCol) * 4;
    d_out[outIdx + 0] = (unsigned char)CLAMP_0_255((int)rVal);
    d_out[outIdx + 1] = (unsigned char)CLAMP_0_255((int)gVal);
    d_out[outIdx + 2] = (unsigned char)CLAMP_0_255((int)bVal);
    int centerIdx = (inRow * width + inCol) * 4 + 3;
    d_out[outIdx + 3] = d_in[centerIdx];
}

static void gpuConvolve(const unsigned char* h_in,
                        unsigned char*       h_out,
                        unsigned width,
                        unsigned height,
                        int threadsPerBlock)
{
    size_t inBytes  = (size_t)width * height * 4;
    size_t outBytes = (size_t)(width - 2) * (height - 2) * 4;
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  inBytes);
    cudaMalloc(&d_out, outBytes);
    cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice);
    int totalInner = (width - 2) * (height - 2);
    int blockSize  = threadsPerBlock;
    int gridSize   = (totalInner + blockSize - 1) / blockSize;
    gpuConvolveKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

static void handleCudaError(cudaError_t err, const char* msg)
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        printf("Usage: %s <input.png> <output.png> <#threads>\n", argv[0]);
        return 1;
    }
    const char* inputFile  = argv[1];
    const char* outputFile = argv[2];
    int threads = atoi(argv[3]);
    if(threads <= 0)
    {
        printf("Error: #threads must be > 0\n");
        return 1;
    }
    float host_w[3][3] = {
        { w[0][0], w[0][1], w[0][2] },
        { w[1][0], w[1][1], w[1][2] },
        { w[2][0], w[2][1], w[2][2] }
    };
    cudaError_t err = cudaMemcpyToSymbol(d_w, host_w, 9 * sizeof(float));
    handleCudaError(err, "cudaMemcpyToSymbol failed for d_w");
    unsigned width = 0, height = 0;
    unsigned char* imageData = nullptr;
    unsigned error = lodepng_decode32_file(&imageData, &width, &height, inputFile);
    if(error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }
    if(width < 3 || height < 3)
    {
        printf("Error: image too small for 3x3 convolution.\n");
        free(imageData);
        return 1;
    }
    unsigned outW = width - 2;
    unsigned outH = height - 2;
    size_t outSize = (size_t)outW * outH * 4;
    unsigned char* outData = (unsigned char*)malloc(outSize);
    if(!outData)
    {
        printf("Error: could not allocate outData\n");
        free(imageData);
        return 1;
    }
    GpuTimer timer;
    timer.Start();
    gpuConvolve(imageData, outData, width, height, threads);
    timer.Stop();
    printf("GPU Convolution Time (%d threads): %.3f ms\n", threads, timer.Elapsed());
    error = lodepng_encode32_file(outputFile, outData, outW, outH);
    if(error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        free(imageData);
        free(outData);
        return 1;
    }
    printf("Output written to: %s\n", outputFile);
    free(imageData);
    free(outData);
    return 0;
}
