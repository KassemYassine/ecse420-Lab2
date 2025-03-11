#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cputimer.h"
#include "lodepng.h"
#include "wm.h"

#define CLAMP_0_255(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x)))

static void handle_error(unsigned error)
{
    if (error)
    {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }
}

static void cpuConvolve(const unsigned char* input,
                        unsigned char*       output,
                        unsigned            width,
                        unsigned            height)
{
    for (unsigned row = 1; row < height - 1; row++)
    {
        for (unsigned col = 1; col < width - 1; col++)
        {
            float rVal = 0.f;
            float gVal = 0.f;
            float bVal = 0.f;
            for (int ii = 0; ii < 3; ii++)
            {
                for (int jj = 0; jj < 3; jj++)
                {
                    int rr = row + (ii - 1);
                    int cc = col + (jj - 1);
                    unsigned inIdx = (rr * width + cc) * 4;
                    rVal += input[inIdx + 0] * w[ii][jj];
                    gVal += input[inIdx + 1] * w[ii][jj];
                    bVal += input[inIdx + 2] * w[ii][jj];
                }
            }
            unsigned outRow = row - 1;
            unsigned outCol = col - 1;
            unsigned outIdx = (outRow * (width - 2) + outCol) * 4;
            output[outIdx + 0] = (unsigned char)CLAMP_0_255((int)rVal);
            output[outIdx + 1] = (unsigned char)CLAMP_0_255((int)gVal);
            output[outIdx + 2] = (unsigned char)CLAMP_0_255((int)bVal);
            unsigned centerIdx = (row * width + col) * 4 + 3;
            output[outIdx + 3] = input[centerIdx];
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <input.png> <output.png>\n", argv[0]);
        return 1;
    }
    const char* inputFile  = argv[1];
    const char* outputFile = argv[2];
    unsigned char* imageData = nullptr;
    unsigned width = 0, height = 0;
    unsigned error = lodepng_decode32_file(&imageData, &width, &height, inputFile);
    if (error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }
    if (width < 3 || height < 3)
    {
        printf("Error: image too small for 3x3 convolution\n");
        free(imageData);
        return 1;
    }
    unsigned outW = width - 2;
    unsigned outH = height - 2;
    size_t outSize = (size_t)outW * outH * 4;
    unsigned char* outData = (unsigned char*)malloc(outSize);
    if (!outData)
    {
        printf("Error: couldn't allocate output memory\n");
        free(imageData);
        return 1;
    }
    CpuTimer cpuTimer;
    cpuTimer.Start();
    cpuConvolve(imageData, outData, width, height);
    cpuTimer.Stop();
    double elapsedMs = cpuTimer.Elapsed();
    printf("CPU Convolution Time: %.3f ms\n", elapsedMs);
    error = lodepng_encode32_file(outputFile, outData, outW, outH);
    if (error)
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
