# B. Finite Element Music Synthesis

## 1. Sequential 4x4 Impementation
- Compile: `make sequential`
- Run: `.\grid_4_4_seq.exe <number of iterations>`

## 2. Parallel 4x4 Implementation
- Compile: `make parallel`
- Run: `.\grid_4_4.exe <number of iterations>`

## 3. Parallel 512x512 implementation
The idea behind this implementation is to divide the main grid into subgrids:
- **Outer grid:** A square division of the main grid into different sets of nodes. Each thread operates on a single element of the outer grid.
- **Inner grid:** An element of the outer grid, containing a set of nodes. It must have a square shape.

Conditions:
- The total number of threads must be less than the number of nodes in the main grid. Thus, `2x2` is the smallest inner grid that a thread can handle (i.e. 4 nodes).

Usage:
- Compile: `make parallel_512`
- Run: `.\grid_512_512.exe <number of iterations>`