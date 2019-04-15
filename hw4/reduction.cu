/**
File: reduction.cu
Author: Terrence Alsup
Date: April 15, 2019
HPC 2019 : HW 4

Implement dot product and matrix-vector product in CUDA.
**/
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


/**
Compute the dot product of vectors a and b of length N and store the result
at c location.  Computation is done on the CPU.
**/
void dotprod(long N, const double* a, const double* b, double* c) {
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for(long i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  *c = sum;
}


/**
Compute the matrix-vector product of matrix A and vector b of dimensions NxN and
N.  Store the result in the vector c.  Computation is done on the CPU.
**/
void MvMult(long N, double* A, double* b, double* c) {
  #pragma omp parallel for
  for(long i = 0; i < N; i++) {
    double sum = 0;
    for(long j = 0; j < N; j++) {
        sum += A[i*N + j] * b[j];
    }
    c[i] = sum;
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

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
The kernel for the dot product.
**/
__global__ void dotprod_kernel(double *sum, const double *a, const double *b, long N) {
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx]; // Now multiply a[i] * b[i].
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
The kernel for the maxtrix-vector product Av = b.
**/
__global__ void matvec_kernel(double *b, const double *A, const double *v, long N) {
  __shared__ double smem[BLOCK_SIZE];

  smem[threadIdx.x] = 0;
  for (long t = threadIdx.x; t < N; t += BLOCK_SIZE ) {
    // Recall that blockIdx.x just corresponds to the row.
    smem[threadIdx.x] += A[blockIdx.x * N + t] * v[t];
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
    if (threadIdx.x == 0) b[blockIdx.x] = smem[0] + smem[1];
  }
}


/**
Compute the dot product <a,b> = sum using the GPU.
**/
void gpu_dot(long N, double *a, double *b, double *sum) {

  // Allocate space on the GPU.
  double *a_d, *b_d;
  cudaMalloc(&a_d, N * sizeof(double));
  cudaMalloc(&b_d, N * sizeof(double));


  // Transfer data to GPU.
  cudaMemcpyAsync(a_d, a, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(b_d, b, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();


  double *y_d; // Store results from blocks.
  long N_work = 1; // Number of blocks we will need.
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&y_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks


  double tt = omp_get_wtime(); // Record the time.

  // Compute the number of blocks we need.
  double *sum_d = y_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);

  // Compute the dot product for each block.
  dotprod_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, a_d, b_d, N);

  // Now add the results from all of the blocks.
  while (Nb > 1) {
    long N_temp = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d + N_temp, sum_d, N_temp);
    sum_d += N_temp;
  }

  // Transfer result back to CPU.
  cudaMemcpyAsync(sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double elapsed = omp_get_wtime() - tt;
  printf("Vector-Vector Multiply GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / elapsed/1e9);


  // Free the memory allocated on the GPUs.
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(y_d);
}

/**
Compute the matrix vector product Ax = b using the GPU.
**/
void gpu_matvec(long N, double *A, double *x, double *b) {
  // Allocate space on the GPU.
  double *A_d, *x_d;
  cudaMalloc(&A_d, N * N * sizeof(double));
  cudaMalloc(&x_d, N * sizeof(double));



  // Transfer data to GPU.
  cudaMemcpyAsync(A_d, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  double *b_d; // Store results for each entry.
  cudaMalloc(&b_d, N * sizeof(double));

  double tt = omp_get_wtime(); // Record the time.


  // Compute the matrix-vector product using the GPU.
  // Note that since A has size N-by-N we use N as the number of blocks.
  matvec_kernel<<<N, BLOCK_SIZE>>>(b_d, A_d, x_d, N);


  // Transfer result back to CPU.
  cudaMemcpyAsync(b, b_d, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double elapsed = omp_get_wtime() - tt;
  printf("Matrix-Vector Multiply GPU Bandwidth = %f GB/s\n", (3*N + N*N) * sizeof(double) / (elapsed*1e9));

  // Free the memory allocated on the GPUs.
  cudaFree(A_d);
  cudaFree(x_d);
  cudaFree(b_d);
}


int main() {
  // Length of the vectors and size of the matrix.
  long N = (1UL<<10); // N = 2^10

  // Declare and allocate space on the CPU for the vectors and matrix as well
  // as the reference solution of the matrix-vector multiplication and the GPU
  // computed solution.
  double *v1, *v2, *A, *prod_ref, *prod;
  cudaMallocHost((void**)&v1, N * sizeof(double));
  cudaMallocHost((void**)&v2, N * sizeof(double));
  cudaMallocHost((void**)&A, N * N * sizeof(double));
  cudaMallocHost((void**)&prod_ref, N * sizeof(double));
  cudaMallocHost((void**)&prod, N * sizeof(double));


  // Initialize the vectors.
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    v1[i] = drand48();
    v2[i] = drand48();
    prod_ref[i] = 0;
    prod[i] = 0;
  }
  // Initialize the matrix.
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N * N; i++) {
    A[i] = drand48();
  }

  // Compute the reference solution for the dot product on the CPU.
  double dot_ref = 0;
  double tt = omp_get_wtime();
  dotprod(N, v1, v2, &dot_ref);
  double elapsed = omp_get_wtime() - tt;
  printf("\nVector-Vector Multiply CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / elapsed/1e9);

  // Now compute the dot product on the GPU.
  double dot = 0;
  gpu_dot(N, v1, v2, &dot);
  printf("Vector-Vector Multiply Error = %f\n\n", fabs(dot-dot_ref));

  // Compute the reference solution for the matrix-vector product on the CPU.

  cudaMallocHost((void**)&prod_ref, N * sizeof(double));
  tt = omp_get_wtime();
  MvMult(N, A, v1, prod_ref);
  elapsed = omp_get_wtime() - tt;
  printf("Matrix-Vector Multiply CPU Bandwidth = %f GB/s\n", (3*N + N*N) * sizeof(double) / elapsed/1e9);

  // Compute the matrix-vector multiplication on the GPU.
  gpu_matvec(N, A, v1, prod);

  // Compute the L2 error of the vectors.
  double error = 0;
  #pragma omp parallel for reduction(+:error)
  for (long i = 0; i < N; i++) {
    error += (prod[i] - prod_ref[i]) * (prod[i] - prod_ref[i]);
  }
  printf("Matrix-Vector Multiply Error = %f\n\n", error);


  // Free all the memory now.
  cudaFreeHost(v1);       // The first vector.
  cudaFreeHost(v2);       // The second vector.
  cudaFreeHost(A);        // The matrix.
  cudaFreeHost(prod_ref); // The vector that contained the product.
  cudaFreeHost(prod);     // The vector of the product computed with the GPU.

  return 0;
}
