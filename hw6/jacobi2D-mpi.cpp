/* MPI-parallel Jacobi smoothing to solve -u''=f on the unit square.
 * Global vector has N unknowns, each processor works with its
 * part, which has lN^2 = N/p unknowns.
 * Author: Terrence Alsup
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i, j;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++) {
    for (j = 1; j <= lN; j++) {
      tmp = 4 * lu[(lN + 2) * i + j];
      tmp -= lu[(lN + 2) * (i - 1) + j] + lu[(lN + 2) * (i + 1) + j];
      tmp -= lu[(lN + 2) * i + j - 1] + lu[(lN + 2) * i + j + 1];
      tmp = invhsq * tmp - 1;
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  MPI_Status status0, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  // Note that log4p is called j in the homework pdf file.
  int log4p = log(p) / log(4);
  int rootp = pow(2, log4p); // Number of processors for each row and column.
  lN = N / rootp;
  // Check the number of processors is correct.
  if (pow(4, log4p) != p && mpirank == 0 ) {
    printf("p: %d\n", p);
    printf("Exiting. p must be a power of 4.\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  if (lN * pow(2, log4p) != N && mpirank == 0) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be divisible by sqrt(p).\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }



  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double * lutemp;
  double * buffer = (double *) malloc(sizeof(double) * (lN + 2));

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    for (int i = 1; i <= lN; i++) {
      for (int j = 1; j <= lN; j++) {
        // Note that f = 1.
        lunew[(lN + 2) * i + j]  = 0.25 * (hsq + lu[(lN + 2) * (i - 1) + j]
                                               + lu[(lN + 2) * (i + 1) + j]
                                               + lu[(lN + 2) * i + j - 1]
                                               + lu[(lN + 2) * i + j + 1]);
      }
    }


    /* communicate ghost values */

    //  We are not on the top edge.

    if (mpirank / rootp < rootp - 1) {
      // Send and recieve values from above.

      // Values to send.
      for (int i = 0; i <= lN + 1; i++) {
        buffer[i] = lunew[(lN + 2) * lN + i];
      }
      MPI_Send(buffer, lN + 2, MPI_DOUBLE, mpirank + rootp, 101, MPI_COMM_WORLD);

      // Values to recieve.
      MPI_Recv(buffer, lN + 2, MPI_DOUBLE, mpirank + rootp, 102, MPI_COMM_WORLD, &status0);
      for (int i = 0; i <= lN + 1; i++) {
        lunew[(lN + 2) * (lN + 1) + i] = buffer[i];
      }
    }

    //  We are not on the bottom edge.
    if (mpirank / rootp > 0) {
      // Send and recieve values from below.

      // Values to send.
      for (int i = 0; i <= lN + 1; i++) {
        buffer[i] = lunew[lN + i];
      }
      MPI_Send(buffer, lN + 2, MPI_DOUBLE, mpirank - rootp, 102, MPI_COMM_WORLD);

      // Values to recieve.
      MPI_Recv(buffer, lN + 2, MPI_DOUBLE, mpirank - rootp, 101, MPI_COMM_WORLD, &status1);
      for (int i = 0; i <= lN + 1; i++) {
        lunew[i] = buffer[i];
      }
    }

    //  We are not on the left edge.
    if (mpirank % rootp > 0) {
      // Send and recieve values from the left.

      // Values to send.
      for (int i = 0; i <= lN + 1; i++) {
        buffer[i] = lunew[(lN + 2) * i + 1];
      }
      MPI_Send(buffer, lN + 2, MPI_DOUBLE, mpirank - 1, 103, MPI_COMM_WORLD);

      // Values to recieve.
      MPI_Recv(buffer, lN + 2, MPI_DOUBLE, mpirank - 1, 104, MPI_COMM_WORLD, &status2);
      for (int i = 0; i <= lN + 1; i++) {
        lunew[(lN + 2) * i] = buffer[i];
      }
    }

    //  We are not on the right edge.
    if (mpirank % rootp < rootp - 1) {
      // Send and recieve values from the right.

      // Values to send.
      for (int i = 0; i <= lN + 1; i++) {
        buffer[i] = lunew[(lN + 2) * i + lN];
      }
      MPI_Send(buffer, lN + 2, MPI_DOUBLE, mpirank + 1, 104, MPI_COMM_WORLD);

      // Values to recieve.
      MPI_Recv(buffer, lN + 2, MPI_DOUBLE, mpirank + 1, 103, MPI_COMM_WORLD, &status3);
      for (int i = 0; i <= lN + 1; i++) {
        lunew[(lN + 2) * i + lN + 1] = buffer[i];
      }
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 1000)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	       printf("Iter %d: Residual: %f\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(buffer);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
