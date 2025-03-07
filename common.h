#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ETA 0.0002f
#define RHO 0.5f
#define G 0.75f

/**
 * @brief Checks if a CUDA call failed. If so, prints the given error message and jumps to goToLabel.
 */
#define checkCudaError(cudaStatus, errorMessage, goToLabel)                     \
  if (cudaStatus != cudaSuccess)                                                \
  {                                                                             \
    fprintf(stderr, "[%s] %s\n", errorMessage, cudaGetErrorString(cudaStatus)); \
    goto goToLabel;                                                             \
  }

#endif