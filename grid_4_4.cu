#include "common.h"
#include "gputimer.h"

/** @brief The size of the grid. Its square cannot exceed the maximum number of threads (1024). */
#define N 4
#if N*N > 1024
    #error "N*N must be less than 1024"
#endif

/**
 * @brief Returns the index of the element at (i, j) in the grid of size N. 
 */
#define idx(gridIndex, row, col) (((gridIndex) * N * N) + ((row) * N) + (col))


/**
 * @brief Simulates a drum sound via the finite element method.
 * 
 * @param iterations - The number of iterations to simulate
 * @param u - The grids containing the two previous states and the current state (to be computed).
 */
__global__ void simulate(int iterations, float* u) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    for (int iter = 0; iter < iterations; iter++) {
        // Compute the current state (interior elements)
        if (i > 0 && i < N-1 && j > 0 && j < N-1) {
            float newValue =
                u[idx(1,i-1,j)] + 
                u[idx(1,i+1,j)] + 
                u[idx(1,i,j-1)] + 
                u[idx(1,i,j+1)] - 
                4*u[idx(1,i,j)];
            newValue *= RHO;
            newValue += 2.0f*u[idx(1,i,j)];
            newValue -= (1-ETA)*u[idx(2,i,j)];
            u[idx(0,i,j)] = newValue / (1+ETA);
        }
        __syncthreads();

        // Compute the current state (boundary elements)
        if (i == 0 && j > 0 && j < N-1) {
            u[idx(0,0,j)] = G * u[idx(0,1,j)];
        }
        else if (i == N-1 && j > 0 && j < N-1) {
            u[idx(0,N-1,j)] = G * u[idx(0,N-2,j)];
        }
        else if (j == 0 && i > 0 && i < N-1) {
            u[idx(0,i,0)] = G * u[idx(0,i,1)];
        }
        else if (j == N-1 && i > 0 && i < N-1) {
            u[idx(0,i,N-1)] = G * u[idx(0,i,N-2)];
        }
        __syncthreads();

        // Compute the current state (corners)
        if (i == 0 && j == 0) 
            u[idx(0,0,0)] = G * u[idx(0,1,0)];
        else if (i == N-1 && j == 0)
            u[idx(0,N-1,0)] = G * u[idx(0,N-2,0)];
        else if (i == 0 && j == N-1)
            u[idx(0,0,N-1)] = G * u[idx(0,0,N-2)];
        else if (i == N-1 && j == N-1)
            u[idx(0,N-1,N-1)] = G * u[idx(0,N-1,N-2)];
        __syncthreads();

        // Shift the grids
        u[idx(2,i,j)] = u[idx(1,i,j)];
        u[idx(1,i,j)] = u[idx(0,i,j)];

        // Print the current state
        if (i == N/2 && j == N/2) {
            printf("(%d,%d): %f\n", i, j, u[idx(0,i,j)]);
        }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    // Validate arguments
    if (argc != 2) {
        printf("Usage: %s <number of iterations>\n", argv[0]);
        return 1;
    }

    // Get the number of iterations from the command line
    int iterations = atoi(argv[1]);

    cudaError_t cudaStatus;
    GpuTimer timer;

    // Allocate memory for the grids
    float* u = (float*)malloc(3 * N * N * sizeof(float));
    
    // Initialize the grids with zeros and set the initial condition
    for (int i = 0; i < 3 * N * N; i++) {
        u[i] = 0.0f;
    }
    u[idx(1,N/2,N/2)] = 1.0f;

    // Allocate memory for the grids on the device
    float* d_u = NULL;
    cudaStatus = cudaMalloc(&d_u, 3 * N * N * sizeof(float));
    checkCudaError(cudaStatus, "Failed to allocate memory on the device", Error);

    // Copy the grids to the device
    cudaStatus = cudaMemcpy(d_u, u, 3 * N * N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(cudaStatus, "Failed to copy memory to the device", Error);

    // Simulate the drum sound
    dim3 threadsPerBlock(N, N);
    timer.Start();
    simulate<<<1, threadsPerBlock>>>(iterations, d_u);
    cudaStatus = cudaGetLastError();
    checkCudaError(cudaStatus, "Failed to launch kernel", Error);
    cudaStatus = cudaDeviceSynchronize();
    checkCudaError(cudaStatus, "Failed to synchronize device", Error);
    timer.Stop();

    // Copy the grids from the device
    cudaStatus = cudaMemcpy(u, d_u, 3 * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(cudaStatus, "Failed to copy memory from the device", Error);

    // Print the elapsed time
    printf("Elapsed time: %f ms\n", timer.Elapsed());

Error:
    // Free memory
    free(u);
    cudaFree(d_u);
    
    return 0;
}