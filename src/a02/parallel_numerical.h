//
// Implements parallel solvers.
//

#ifndef HPSC_PARALLEL_NUMERICAL_H
#define HPSC_PARALLEL_NUMERICAL_H

#include "mpi.h"

#include "matrix.h"
#include "serial_numerical.h"

#define MPI_SETUP  \
  int local_id, n_procs; \
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id); \
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#define MASTER if (0 == local_id)

Matrix<double> SolveParallel(BandMatrix<double> &A, Matrix<double> &b, int argc, char **argv) {
  MPI_SETUP;

  MASTER {
    cout << "Running parallel solver on " << n_procs << " processors." << endl;
    double *data = new double(b.rows_);
  }
  else {

  }

  // TODO solve stuff in parallel
  MASTER {
    cout << "Actually just doing it on main processor for debug." << endl;
  };

  return SolveSerial(A, b, true);
}

#endif //HPSC_PARALLEL_NUMERICAL_H
