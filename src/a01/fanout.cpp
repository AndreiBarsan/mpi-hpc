/// A simple test implementation of one-to-all broadcast.

#include <cmath>
#include "mpi.h"
#include "src/common/utils.h"

/// Returns the ith bit of n.
bool bit(unsigned int i, unsigned int n) {
  return ((n >> i) & 1U) == 1U;
}

int fanout_experiment(int argc, char **argv) {
  using namespace std;
  unsigned int npin = 0;

  MPI_Init(&argc, &argv);
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);


  if (! is_power_of_two(n_procs)) {
    throw runtime_error("The number of processors must be a power of two in a hypercube!");
  }

  auto d = static_cast<int>(log2(n_procs));

  int val = -1;
  if (local_id == npin) {
    val = 13;
  }

  // TODO(andreib): Implement dis shit for arbitrary input processor.
  for (unsigned int i = 0; i < d; i++) {
    if (local_id < (1 << (i + 1))) {
//      cout << "Step " << i << ", id = " << local_id << " active!" << endl;

      MPI_Status status;
      int rec_src = Flip(i, local_id);
      if (rec_src < local_id) {
        MPI_Recv(&val, 1, MPI_INT, rec_src, 0, MPI_COMM_WORLD, &status);
      }
      else {
        MPI_Send(&val, 1, MPI_INT, rec_src, 0, MPI_COMM_WORLD);
      }
    }
  }

  cout << local_id << ": " << val << endl;

  MPI_Finalize();
  return 0;
}


int main(int argc, char **argv) {
  return fanout_experiment(argc, argv);
}
