#include "mpi_eigen_helpers.h"

#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include "common/mpi_helpers.h"

// TODO(andreib): Send all sizes as MPI_LONG!
// TODO-LOW(andreib): Use single send with *void data / structs in MPI.
// TODO-LOW(andreib): Perform an asynchronous broadcast.

void BroadcastEigenSparse(ESMatrix &A, int sender) {
  MPI_SETUP;
  if (! A.isCompressed()) {
    throw std::runtime_error("Can only broadcast a sparse matrix with compressed storage!");
  }
  int i_buffer[3];
  int total_element_count = -1;
  if (local_id == sender) {
    total_element_count = static_cast<int>(A.nonZeros());
    i_buffer[0] = static_cast<int>(A.rows());
    i_buffer[1] = static_cast<int>(A.cols());
    i_buffer[2] = total_element_count;

//    int rows = i_buffer[0], cols = i_buffer[1]; // for debugging
//    assert(rows==A.innerSize() && cols==A.outerSize());
//    assert(A.outerIndexPtr()[i_buffer[1]]==total_element_count);
  }
  MPI_Bcast(i_buffer, 3, MPI_INT, sender, MPI_COMM_WORLD);
  int n_rows = i_buffer[0];
  int n_cols = i_buffer[1];
  total_element_count = i_buffer[2];

  if (local_id != sender) {
//    cout << local_id << ": resize to " << n_rows << ", " << n_cols << "." << endl;
    A.resize(n_rows, n_cols);
    A.reserve(total_element_count);
  }
  MPI_Bcast(A.valuePtr(),      total_element_count, MPI_DOUBLE, sender, MPI_COMM_WORLD);
  MPI_Bcast(A.innerIndexPtr(), total_element_count, MPI_INT,    sender, MPI_COMM_WORLD);
  MPI_Bcast(A.outerIndexPtr(), n_cols,              MPI_INT,    sender, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (local_id != sender) {
    A.outerIndexPtr()[n_cols] = total_element_count;
  }
}

void AllToAllEigenDense(const EMatrix &in_chunk, EMatrix &out) {
  MPI_SETUP;
  using namespace Eigen;
  using namespace std;
  cout << local_id << ": All to all, in chunk size" << in_chunk.rows() << " x " << in_chunk.cols() << "\n";

  int idx = 0;
  int tile_size = in_chunk.rows();
  cout << in_chunk.rows() << ", " << in_chunk.cols() << ", " << n_procs << endl;
  assert(in_chunk.rows() * n_procs == in_chunk.cols());
  int n_els = tile_size * tile_size * n_procs;

  // TODO(andreib): We're doing all-to-all, not allgather; why do we need this extra memory?
  auto send_buffer = make_unique<double[]>(n_els * n_procs);
  auto recv_buffer = make_unique<double[]>(n_els * n_procs);
//  memset(recv_buffer, 0, n_els * n_procs * sizeof(double)); // TODO get rid of this?
  for(int k = 0; k < n_procs; ++k) {
    for(int i = 0; i < tile_size; ++i) {
      for(int j = 0; j < tile_size; ++j) {
        // Flipped j and i since we are transposing each chunk.
        send_buffer[idx++] = in_chunk(j, k * tile_size + i);
      }
    }
  }
//  cout << in_chunk << endl;
  MPI_Alltoall(
      send_buffer.get(),
      tile_size * tile_size,
      MPI_DOUBLE,
      recv_buffer.get(),
      tile_size * tile_size,
      MPI_DOUBLE,
      MPI_COMM_WORLD);
//  cout << "All to all successful!" << endl;

  idx = 0;
  out.resize(in_chunk.rows(), in_chunk.cols());
  for(int k = 0; k < n_procs; ++k) {
    for(int i = 0; i < tile_size; ++i) {
      for(int j = 0; j < tile_size; ++j) {
        out(i, k * tile_size + j) = recv_buffer[idx++];
      }
    }
  }
}
