/*
Terrence Alsup
HPC Spring 2019
March 13, 2019

Debugged code for Open MP.
The main change was to simply use malloc to create space on the heap.  N = 1048
was too large to fit on the stack, which was causing the segmentation fault.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[])
{
int nthreads, tid, i, j;
double **a;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {
  // Allocate space on the heap for the matrix.
  // The rows of the columns of the matrix.
  a = (double**) malloc(N*sizeof(double*));

  // Allocate space for each column.
  for( i = 0; i < N; i++) {
    // Space for the individual entries in a column.
    a[i] = (double*) malloc(N*sizeof(double));
  }


  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);

  }  /* All threads join master thread and disband */

}
