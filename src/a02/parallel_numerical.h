//
// Implements parallel solvers.
//

#ifndef HPSC_PARALLEL_NUMERICAL_H
#define HPSC_PARALLEL_NUMERICAL_H

#include "serial_numerical.h"

#define MPI_SETUP  \
  int local_id, n_procs; \
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id); \
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#define MASTER if (0 == local_id)

#ifdef DEBUG_WITH_EIGEN
/// Debug serial implementation of parallel partitioning method II.
Matrix<double> SolveParallelDebug(const BandMatrix<double> &A_raw, const Matrix<double> &b_raw);
#endif    // DEBUG_WITH_EIGEN

/// Solves the given banded system Ax = b using MPI and the parallel partitioning method II.
/// \param A An n x n banded matrix.
/// \param b An n x 1 vector.
/// \return The n x 1 solution vector.
Matrix<double> SolveParallel(BandMatrix<double> &A, Matrix<double> &b);


#endif //HPSC_PARALLEL_NUMERICAL_H
