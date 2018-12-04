#include "mpi_eigen_helpers.h"

#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include "common/mpi_helpers.h"

// TODO(andreib): Send all sizes as MPI_LONG!
// TODO-LOW(andreib): Use single send with *void data / structs in MPI.
// TODO-LOW(andreib): Perform an asynchronous broadcast.

using namespace Eigen;
using namespace std;

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

void TransposeEigenDense(const EMatrix &in_chunk, EMatrix &out) {
  MPI_SETUP;
  int idx = 0;
  assert(in_chunk.rows() * n_procs == in_chunk.cols());
  long tile_rows = in_chunk.rows();
  long tile_cols = in_chunk.rows();  // We just asserted that the tiles must be square.
  size_t tile_size = tile_rows * tile_cols;
  auto send_buffer = make_unique<double[]>(tile_size * n_procs);
  auto recv_buffer = make_unique<double[]>(tile_size * n_procs);

  for(int k = 0; k < n_procs; ++k) {
    for(int i = 0; i < tile_rows; ++i) {
      for(int j = 0; j < tile_cols; ++j) {
        // We read the data in row-major order, but Eigen stores matrices in column-major (Fortran) order.
        // Thus, when Alltoall does its thing, it writes the row-major data into a column-major Eigen buffer, therefore
        // doing the intra-tile transpose for us!
        send_buffer[idx++] = in_chunk(i, k * tile_cols + j);
      }
    }
  }
  out.resize(in_chunk.rows(), in_chunk.cols());

  // Redistribute the chunks such that processor #1 gets the 1st chunk in each processor, processor #2 gets the 2nd
  // chunk in each processor, etc., thereby effectively performing a block-wise transpose.
  MPI_Alltoall(send_buffer.get(), tile_size, MPI_DOUBLE, out.data(), tile_size, MPI_DOUBLE, MPI_COMM_WORLD);
}
