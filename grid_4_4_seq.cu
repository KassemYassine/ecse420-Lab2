#include "common.h"
#include "cputimer.h"

#define N 4
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
void simulate(int iterations, float* u) {
    for (int iter = 0; iter < iterations; iter++) {
        // Compute the current state (interior elements)
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
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
        }

        // Compute the current state (boundary elements)
        for (int i = 1; i < N-1; i++) {
            u[idx(0,0,i)] = G * u[idx(0,1,i)];
            u[idx(0,N-1,i)] = G * u[idx(0,N-2,i)];
            u[idx(0,i,0)] = G * u[idx(0,i,1)];
            u[idx(0,i,N-1)] = G * u[idx(0,i,N-2)];
        }

        // Compute the current state (corners)
        u[idx(0,0,0)] = G * u[idx(0,1,0)];
        u[idx(0,N-1,0)] = G * u[idx(0,N-2,0)];
        u[idx(0,0,N-1)] = G * u[idx(0,0,N-2)];
        u[idx(0,N-1,N-1)] = G * u[idx(0,N-1,N-2)];

        // Print the current state
        printf("(%d,%d): %f\n", N/2, N/2, u[idx(0,N/2,N/2)]);

        // Shift the grids
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                u[idx(2,i,j)] = u[idx(1,i,j)];
                u[idx(1,i,j)] = u[idx(0,i,j)];
            }
        }
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

    // Print grid size
    printf("Grid size: %d\n", N);

    CpuTimer timer;

    // Allocate memory for the grids
    float* u = (float*)malloc(3 * N * N * sizeof(float));
    
    // Initialize the grids with zeros and set the initial condition
    for (int i = 0; i < 3 * N * N; i++) {
        u[i] = 0.0f;
    }
    u[idx(1,N/2,N/2)] = 1.0f;

    // Simulate the drum sound
    timer.Start();
    simulate(iterations, u);
    timer.Stop();

    // Print the elapsed time
    printf("Elapsed time: %f ms\n", timer.Elapsed());

    // Free memory
    free(u);
    
    return 0;
}