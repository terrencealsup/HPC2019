// Terrence Alsup
#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_pingpong(int rank, int size, long Nrepeat, long Nsize, MPI_Comm comm) {

  int* msg = (int*) malloc(Nsize * sizeof(int));

  for (long i = 0; i < Nsize; i++) {
    msg[i] = 0; // Initialize the array to 0.
  }

  int src  = (rank - 1 + size) % size;  // Who is delivering the message?
  int dest = (rank + 1) % size;  // Where are we sending the message to?

  MPI_Status status;
  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  for (long repeat  = 0; repeat < Nrepeat; repeat++) {

    // Send the initial message from process 0 to process 1.
    if (rank == 0 && repeat == 0) MPI_Send(msg, Nsize, MPI_INT, 1, 999, comm);

    MPI_Recv(msg, Nsize, MPI_INT, src, 999, comm, &status);

    // Update each element by its rank.
    for (long j = 0; j < Nsize; j++) {
      msg[j] += rank;
    }

    // Send the message updated by the rank.
    MPI_Send(msg, Nsize, MPI_INT, dest, 999, comm);

  }

  if (rank == 1) MPI_Recv(msg, Nsize, MPI_INT, 0, 999, comm, &status);

  MPI_Barrier(comm);

  tt = MPI_Wtime() - tt;

  // Now compute the error from the expected value.
  long true_val = Nrepeat * size * (size - 1) / 2;
  if (rank == 0) {
    printf("Error = %d\n", abs(true_val - msg[0]));
  }

  free(msg);
  return tt;
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);


  // Latency, transferring only 1 integer.
  long Nrepeat = 10000;
  double tt = time_pingpong(rank, size, Nrepeat, 1, comm);
  if (!rank) printf("int_ring latency: %e ms\n", tt/Nrepeat * 1000);

  // Bandwidth for a large array of Nsize integers.
  Nrepeat = 10000;
  long Nsize = 1 << 18;
  tt = time_pingpong(rank, size, Nrepeat, Nsize, comm);
  if (!rank) printf("int_ring bandwidth: %e GB/s\n", (Nsize*Nrepeat)/tt/1e9);

  MPI_Finalize();
}
