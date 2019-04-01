/**
Terrence Alsup

Date: April 1st, 2019.
HPC 2019: HW 3

Added the scan_omp method.
**/
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  // The size of the chunks for each thread.
  long chunkSize = n / num_threads;

  #pragma omp parallel
  {
    int threadID = omp_get_thread_num();
    long start = threadID * chunkSize + 1;
    long end;
    // Last chunk has to take care of the rest.
    if (threadID == num_threads - 1) {
      end = n - 1;
    } else {
      end = (threadID + 1) * chunkSize;
    }
    // Sum for each individual chunk.
    for(long i = start; i <= end; i++) {
      prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
    }
  }

  // Now make the correction by combining all of the results done in serial.
  // Note that we do not need to correct the first thread (thread 0).
  for(int t = 1; t < num_threads; t++) {
    long start = t * chunkSize + 1;
    long end;
    // Last chunk had to take care of the rest.
    if (t == num_threads - 1) {
      end = n - 1;
    } else {
      end = (t + 1) * chunkSize;
    }
    for(long i = start; i <= end; i++) {
      // Add the last value from the previous chunk.
      prefix_sum[i] += prefix_sum[start - 1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
