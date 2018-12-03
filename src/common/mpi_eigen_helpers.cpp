#include "mpi_eigen_helpers.h"

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

//  cout << local_id << " method end OK." << endl;
}
