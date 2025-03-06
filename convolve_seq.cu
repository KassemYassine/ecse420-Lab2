#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <string>
#include "cputimer.h"
#include "lodepng.h"
#include "wm.h" // For the 3x3 matrix "w"

#define CLAMP_0_255(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x)))

static void handle_error(unsigned error)
{
    if (error)
    {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }
}

/**
 * @brief Apply a 3x3 convolution, but replicate the border so the output
 *        remains the same size (width x height).
 * 
 * @param input   Pointer to input RGBA image data
 * @param output  Pointer to output RGBA image data
 * @param width   Image width
 * @param height  Image height
 */
static void cpuConvolve(const unsigned char* input,
                        unsigned char*       output,
                        unsigned            width,
                        unsigned            height)
{
    // For each row and column in the output
    for (unsigned row = 0; row < height; row++)
    {
        for (unsigned col = 0; col < width; col++)
        {
            float rVal = 0.f;
            float gVal = 0.f;
            float bVal = 0.f;

            // 3x3 filter
            for (int ii = -1; ii <= 1; ii++)
            {
                for (int jj = -1; jj <= 1; jj++)
                {
                    // Row/col in input for this filter tap,
                    // clamped to the valid [0..height-1], [0..width-1].
                    int rPos = (int)row + ii;
                    int cPos = (int)col + jj;
                    if (rPos < 0) rPos = 0;
                    if (rPos >= (int)height) rPos = height - 1;
                    if (cPos < 0) cPos = 0;
                    if (cPos >= (int)width)  cPos = width - 1;

                    unsigned inIdx = (rPos * width + cPos) * 4;
                    // w[ii+1][jj+1] is the filter weight
                    rVal += input[inIdx + 0] * w[ii + 1][jj + 1];
                    gVal += input[inIdx + 1] * w[ii + 1][jj + 1];
                    bVal += input[inIdx + 2] * w[ii + 1][jj + 1];
                }
            }

            // Write output pixel (clamped)
            unsigned outIdx = (row * width + col) * 4;
            output[outIdx + 0] = (unsigned char)CLAMP_0_255((int)rVal);
            output[outIdx + 1] = (unsigned char)CLAMP_0_255((int)gVal);
            output[outIdx + 2] = (unsigned char)CLAMP_0_255((int)bVal);

            // Preserve alpha from the *original* pixel
            unsigned alphaIdx = (row * width + col) * 4 + 3;
            output[outIdx + 3] = input[alphaIdx];
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

    // 1) Decode directly from file into RGBA 8-bit
    unsigned char* imageData = nullptr;
    unsigned width = 0, height = 0;
    unsigned error = lodepng_decode32_file(&imageData, &width, &height, inputFile);
    if (error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    // 2) Allocate same-size output
    size_t outSize = (size_t)width * height * 4; // RGBA
    unsigned char* outData = (unsigned char*)malloc(outSize);
    if (!outData)
    {
        printf("Error: couldn't allocate output memory\n");
        free(imageData);
        return 1;
    }

    // 3) Run CPU convolution
    CpuTimer cpuTimer;
    cpuTimer.Start();
    cpuConvolve(imageData, outData, width, height);
    cpuTimer.Stop();

    double elapsedMs = cpuTimer.Elapsed();
    printf("CPU Convolution Time: %.3f ms\n", elapsedMs);

    // 4) Encode result to file as RGBA (same w,h)
    error = lodepng_encode32_file(outputFile, outData, width, height);
    if (error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        free(imageData);
        free(outData);
        return 1;
    }

    printf("Output written to: %s\n", outputFile);

    // Clean up
    free(imageData);
    free(outData);

    return 0;
}
