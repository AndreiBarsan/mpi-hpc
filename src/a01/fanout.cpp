/// A simple implementation of one-to-all broadcast for Problem 1 from Assignment 1.

#include <cmath>
#include <iostream>

#include "gflags/gflags.h"
#include "mpi.h"

#include "common/utils.h"

DEFINE_int32(npin, 0, "The processor which is the origin in the fan-out process.");


/// Returns the ith bit of n.
bool Bit(unsigned int i, unsigned int n) {
  return ((n >> i) & 1U) == 1U;
}

int fanout_experiment(int argc, char **argv) {
  using namespace std;
  const int kStartVal = 13;
  const int kNpIn = FLAGS_npin;
  MPI_Init(&argc, &argv);
  int local_id = -1, n_procs = -1;
  int send_buf, recv_buf;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  if (!IsPowerOfTwo(n_procs)) {
    throw runtime_error("The number of processors must be a power of two in a hypercube!");
  }

  if (local_id == 0) {
    cout << "Input is at " << kNpIn << endl;
  }

  auto d = static_cast<int>(log2(n_procs));

  int val = -1;
  if (local_id == kNpIn) {
    val = kStartVal;
  }

  // This works (Method A)
//  for (unsigned int i = 0; i < d; i++) {
//    send_buf = val;
//    MPI_Send(&send_buf, 1, MPI_INT, Flip(i, local_id), 0, MPI_COMM_WORLD);
//    MPI_Status status;
//    MPI_Recv(&recv_buf, 1, MPI_INT, Flip(i, local_id), 0, MPI_COMM_WORLD, &status);
//    if (recv_buf != -1) {
//      val = recv_buf;
//    }
//  }

  // Method B: A little less redundancy
  for (int i = d - 1; i >= 0; i--) {
    if (Bit(i, local_id) == Bit(i, kNpIn)) {
      send_buf = val;
      MPI_Send(&send_buf, 1, MPI_INT, Flip(i, local_id), 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Status status;
      MPI_Recv(&recv_buf, 1, MPI_INT, Flip(i, local_id), 0, MPI_COMM_WORLD, &status);
      val = recv_buf;
    }

  }

  cout << local_id << ": " << val << endl;
  if (val != kStartVal) {
    throw runtime_error(Format("Fanout failed in node %d! Start node was %d.", local_id, kNpIn));
  }

  MPI_Finalize();
  return 0;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return fanout_experiment(argc, argv);
}
