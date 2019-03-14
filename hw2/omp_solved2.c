/*
Terrence Alsup
HPC Spring 2019
March 13, 2019

Debugged code for Open MP.
The main changes were:
1. Declare tid inside the parallel region to avoid a race condition.
2. Declare total inside the parallel region since we are computing a value for
   each individual thread.
3. Remove the second pragma omp for since we have already split into multiple
   threads.
4. Change total from a float to a double for better accuracy.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads;


/*** Spawn parallel region ***/
#pragma omp parallel
  {
  /* Obtain thread number */
  // This should be declared inside the parallel region since each thread has a
  // different number.
  int tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */

  double total = 0.0; // Change float to double for better accuracy and make
                      // private to each thread.

  // We've already split into multiple threads so get rid of the parallel for.
  // We should see that all threads compute the same total.
  for (int i=0; i<1000000; i++)
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
