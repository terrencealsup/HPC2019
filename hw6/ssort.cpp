// Parallel sample sort
//
// Author: Terrence Alsup
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100000;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // Start timing here.
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int * local_splitters = (int *) malloc(sizeof(int) * (p - 1));
  for (int i = 0; i < p - 1; i++) {
    local_splitters[i] = vec[(i + 1) * N / p];
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int * global_splitters;
  if (rank == 0){
    global_splitters = (int *) malloc(sizeof(int) * p * (p - 1));
  }

  MPI_Gather(local_splitters, p - 1, MPI_INT, global_splitters, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if (rank == 0) {
    std::sort(global_splitters, global_splitters + p * (p - 1));
    for (int i = 0; i < p - 1; i++) {
      local_splitters[i] = global_splitters[(i + 1) * (p - 1)];
    }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(local_splitters, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

  int * sdispls  = (int *) malloc(sizeof(int) * p);
  int * num_send = (int *) malloc(sizeof(int) * p);
  sdispls[0] = 0;
  for (int i = 0; i < p - 1; i++) {
    sdispls[i + 1] = std::lower_bound(vec, vec + N, local_splitters[i]) - vec;
    num_send[i] = sdispls[i + 1] - sdispls[i];
  }
  num_send[p - 1] = N - sdispls[p - 1];

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int * rdispls  = (int *) malloc(sizeof(int) * p); // Recieve displacements.
  int * num_recv = (int *) malloc(sizeof(int) * p); // Number expected to recieve.

  MPI_Alltoall(num_send, 1, MPI_INT, num_recv, 1, MPI_INT, MPI_COMM_WORLD);

  rdispls[0] = 0;
  for (int i = 0; i < p - 1; i++) {
    rdispls[i + 1] = rdispls[i] + num_recv[i];
  }

  // Store the recieved data.  There will be local_N numbers.
  int local_N = rdispls[p - 1] + num_recv[p - 1];
  int * local_result = (int *) malloc(sizeof(int) * local_N);

  MPI_Alltoallv(vec, num_send, sdispls, MPI_INT, local_result, num_recv, rdispls, MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(local_result, local_result + local_N);

  // Finish timing.
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - start;
  if (rank == 0) {
    printf("Time to sort N = %d was t = %fs.\n", N, elapsed);
  }

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename, "w+");

  if (NULL == fd) {
    printf("Error opening file.\n");
    return 1;
  }

  for (int i = 0; i < local_N; i++) {
    fprintf(fd, "%d\n", local_result[i]);
  }
  fclose(fd);


  free(vec);
  free(local_splitters);
  if (rank == 0) free(global_splitters);
  free(sdispls);
  free(num_send);
  free(rdispls);
  free(num_recv);
  free(local_result);


  MPI_Finalize();
  return 0;
}
