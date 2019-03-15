/*
gs2D-omp.cpp

Author: Terrence Alsup
HPC 2019 HW 2
Gauss Seidel OpenMP implementation.
*/
#include <stdio.h>
#include "utils.h"
#include <math.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

double* GaussSeidelStep(long N, double h, double* u, double* f);
double calculateResidual(long N, double h, double* u, double* f);



int main(int argc, char** argv) {

  // Set the number of threads.
  omp_set_num_threads(4);

  // The size of the matrix N-by-N.
  long N = 1000;

  // Set the maximum number of iterations.
  long maxiter = 10000;

  // Set the tolerance for the residual, relative to initial residual > 1.
  double tolerance = 1e6;

  double h = 1.0/(N+1); // The spacing between grid-points.

  // Store the RHS f and u_0, note that we store the boundary values as well.
  // Note that f = 1 and we only implement it this way in case we decide to
  // change it later on.
  double *f = (double *) malloc((N+2)*(N+2)*sizeof(double));

  double *u = (double *) malloc((N+2)*(N+2)*sizeof(double));

  for( int i = 0; i < (N+2)*(N+2); i++) {
    u[i] = 0; // Initial u_0 = 0
    f[i] = 1; // RHS = 1
  }

  double res0 = calculateResidual(N, h, u, f); // Initial residual.
  double res = res0;

  long iter = 0; // Keep track of the iteration.
  Timer t;
  t.tic();
  while(iter < maxiter && res > res0/tolerance) {
    // Only print residual every 50 iterations.
    if( iter % 50 == 0){
      printf("Iteration: %d \t Residual: %f. \n", iter, res);
    }
    // Update u.
    u = GaussSeidelStep(N, h, u, f);
    // Calculate the residual.
    res = calculateResidual(N, h, u, f);
    iter++;
  }
  double time = t.toc();

  // Print the final result.
  printf("\n");
  printf("Gauss Seidel Method for %d-by-%d matrix.\n",N,N);
  printf("Ended after %d iterations with residual %f. \n", iter, res);
  printf("Total time : %f. seconds\n",time);
  printf("\n");

  // Free the arrays that we allocated in memory.
  free(u);
  free(f);

  return 0;
}


// Compute a single step in the Gauss Seidel method.
double* GaussSeidelStep(long N, double h, double* u, double* f) {

  double* u_new = (double *) malloc((N+2)*(N+2)*sizeof(double));

  // First update the red points.
  // Remember that u is stored in column major order.
  # pragma omp parallel for
  for( long j = 1; j <= N; j++ ) {
    for( long i = 1; i <= N; i++) {
      // Check that the point is a red point.
      if(fmod(i+j,2) == 0) {
        // u_{i,j} = u[i+j*(N+2)]
        double RHS = pow(h, 2)*f[i+j*(N+2)];
        RHS += u[i-1 + j*(N+2)];
        RHS += u[i + (j-1)*(N+2)];
        RHS += u[i+1 + j*(N+2)];
        RHS += u[i + (j+1)*(N+2)];
        u_new[i + j*(N+2)] = 0.25*RHS;
      }
    }
  }
  // Next update the black points.
  // Remember that u is stored in column major order.
  # pragma omp parallel for
  for( long j = 1; j <= N; j++ ) {
    for( long i = 1; i <= N; i++) {
      // Check that the point is a black point.
      if(fmod(i+j,2) == 1) {
        // u_{i,j} = u[i+j*(N+2)]
        double RHS = pow(h, 2)*f[i+j*(N+2)];
        RHS += u_new[i-1 + j*(N+2)];
        RHS += u_new[i + (j-1)*(N+2)];
        RHS += u_new[i+1 + j*(N+2)];
        RHS += u_new[i + (j+1)*(N+2)];
        u_new[i + j*(N+2)] = 0.25*RHS;
      }
    }
  }
  free(u);
  return u_new;
}



// Compute the residual of |-Laplacian u - f|_2
double calculateResidual(long N, double h, double *u, double *f) {
  double residual = 0;
  // Loop over all interior points.
  // Note that matrices are stored in column major order so it is important to
  // loop over j first and then i.
  # pragma omp parallel for shared(N, h, u, f) reduction(+:residual)
  for(long j = 1; j <= N; j++) {
    for(long i = 1; i <= N; i++) {
      // u_{i,j} = u[i + j*(N+2)]
      double LHS = -u[i - 1 + j*(N+2)];
      LHS += -u[i + (j-1)*(N+2)];
      LHS += 4*u[i + j*(N-2)];
      LHS += -u[i + 1 + j*(N+2)];
      LHS += -u[i + (j+1)*(N+2)];
      residual += pow(LHS-f[i + j*(N+2)], 2);
    }
  }
  return sqrt(residual);
}
