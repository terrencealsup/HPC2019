#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int rank; // Which process is running this.
  int size; // Get the number of processes.
  int N = 100; // How many times to loop over all processors.


  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Loop over all of the processors.
  for (int i = 0; i < size; i++) {


  }
  if (rank == 0) {
    int message_out = 42;
    int message_in;
    MPI_Status status;

    MPI_Recv(&message_in,  1, MPI_INT, 2, 999, MPI_COMM_WORLD, &status);
    MPI_Send(&message_out, 1, MPI_INT, 2, 999, MPI_COMM_WORLD);

    printf("Rank %d received %d\n", rank, message_in);
  } else if (rank == 2) {
    int message_out = 80;
    int message_in;
    MPI_Status status_2;

    MPI_Send(&message_out, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
    MPI_Recv(&message_in,  1, MPI_INT, 0, 999, MPI_COMM_WORLD, &status_2);

    printf("Rank %d received %d\n", rank, message_in);
  }

  MPI_Finalize();

  return 0;
}
