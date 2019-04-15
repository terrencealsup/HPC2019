/**
File: jacobi2D.cu
Author: Terrence Alsup
Date: April 15, 2019
HPC 2019 : HW 4

Solve the Equation -Lu = f where L is the Laplacian using the Jacobi method.
u = 0 on the boundary
f = 1
**/
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <math.h>

#define BLOCK_SIZE 1024

/**
The reduction kernel to add a vector of numbers in a block.
**/
__global__ void reduction_kernel(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

/**
The kernel to compute the Jacobi update at a site.
**/
__global__ void jacobi_kernel(double *unew, const double *u, const double *f, long N, double h2) {

  // The index of the site on the grid we are updating.
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  long i = idx / (N+2); // The row.
  long j = idx % (N+2); // The column.
  double temp;
  if (i > 0 && i < (N+2) && j > 0 && j < (N+2)) {
    // Only update the interior points.
    temp = h2*f[idx];
    temp += u[(i-1)*(N+2) + j];
    temp += u[i*(N+2) + j - 1];
    temp += u[(i+1)*(N+2) + j];
    temp += u[i*(N+2) + j + 1];
    unew[idx] = 0.25 * temp;
  } else {
    // Boundary points are set to 0 still.
    unew[idx] = 0;
  }
}

/**
Compute the residual |-Du - f|_2^2 for a block of threads.
**/
__global__ void residual_kernel(double *res, const double *u, const double *f, long N, double h2) {
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  long i = idx / (N+2); // The row.
  long j = idx % (N+2); // The column.
  double Du;
  if (i > 0 && i < (N+2) && j > 0 && j < (N+2)) {
    // Discrete Laplacian of u.
    Du = -4 * u[idx];
    Du += u[(i-1)*(N+2) + j];
    Du += u[(i+1)*(N+2) + j];
    Du += u[i*(N+2) + j - 1];
    Du += u[i*(N+2) + j + 1];
    Du /= h2;
    smem[threadIdx.x] = (Du + f[idx]) * (Du + f[idx]); // Residual at the point.
  } else {
    smem[threadIdx.x] = 0;
  }

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) res[blockIdx.x] = smem[0] + smem[1];
  }
}




int main(int argc, char** argv) {

  // Number of interior points along each dimension.
  long N = (1UL<<7); // N = 2^7

  // Arrays for the solution u and RHS f.
  double *u, *f;
  // Allocate space on the CPU.
  // Notice that we include the boundaries as well.
  cudaMallocHost((void**)&u, (N+2) * (N+2) * sizeof(double));
  cudaMallocHost((void**)&f, (N+2) * (N+2) * sizeof(double));

  // Initialize the initial condition for u and f, done in parallel.
  #pragma omp parallel for
  for (long i = 0; i < (N+2)*(N+2); i++) {
    u[i] = 0;
    f[i] = 1;
  }

  // Allocate space on the GPU to calculate the solutions.
  double *u_d, *f_d, *unew_d;
  cudaMalloc(&u_d, (N+2) * (N+2) * sizeof(double));
  cudaMalloc(&f_d, (N+2) * (N+2) * sizeof(double));
  cudaMalloc(&unew_d, (N+2) * (N+2) * sizeof(double));

  // Transfer data to the GPU.
  cudaMemcpyAsync(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(unew_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Now do the Jacobi iterations.
  long maxiter = 1e4;
  double tolerance = 1e-1;          // Absolute tolerance
  double residual = 2 * tolerance;  // Residual |Du-f|_2
  double h2 = 1.0/((N+1) * (N+1));  // Grid size h^2.

  // Nb is number of blocks to compute with for Jacobi on the GPU.
  long Nb = ((N+2)*(N+2) + BLOCK_SIZE - 1) / (BLOCK_SIZE);

  // Extra memory buffer for reduction across thread-blocks.
  double *y_d;
  long N_work = 1;
  for (long i = Nb; i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&y_d, N_work*sizeof(double));

  double start = omp_get_wtime(); // Get the starting time.

  long iter = 1;
  while (iter <= maxiter && residual > tolerance) {

    // Do the Jacobi update on the GPU.
    // Each block will compute the update of a single entry on the grid.
    jacobi_kernel<<<Nb, BLOCK_SIZE>>>(unew_d, u_d, f_d, N, h2);

    // First synchronize all threads and then update u on the GPU.
    cudaDeviceSynchronize();
    cudaMemcpy(u_d, unew_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToDevice);

    // Compute the residual on the GPU.
    double* res_d = y_d; // Result from each block.
    residual_kernel<<<Nb, BLOCK_SIZE>>>(res_d, u_d, f_d, N, h2);

    // Now add the residuals from all of the blocks.
    long Nbres = Nb;
    while (Nbres > 1) {
      long temp = Nbres;
      Nbres = (Nbres+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel<<<Nbres,BLOCK_SIZE>>>(res_d + Nbres, res_d, temp);
      res_d += Nbres;
    }
    // Transfer the computation of the residual to the CPU.
    cudaMemcpyAsync(&residual, res_d, 1*sizeof(double), cudaMemcpyDeviceToHost);

    // Print what the residual was every 500 iterations.
    if (iter % 500 == 0) {
      printf("Residual at Iteration %d = %f\n", iter, residual);
    }

    iter++;
  }
  double elapsed = omp_get_wtime() - start;
  if (residual < tolerance) {
    printf("\nConverged in %d iterations with residual = %f\n", iter, residual);
  } else {
    printf("\nFailed to converge in %d iterations.  Final residual = %f\n", maxiter, residual);
  }
  printf("Total wall clock time (s) = %f\n", elapsed);

  // Now free the memory that was allocated on the CPU and GPU.
  cudaFree(u_d);
  cudaFree(f_d);
  cudaFree(unew_d);
  cudaFree(y_d);
  cudaFreeHost(u);
  cudaFreeHost(f);

  return 0;
}
