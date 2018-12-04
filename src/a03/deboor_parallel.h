/**
 * @file deboor_parallel.h
 * @brief Functions for solving linear systems in parallel using DeBoor decomposition.
 */

#ifndef HPSC_DEBOOR_PARALLEL_H
#define HPSC_DEBOOR_PARALLEL_H

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "common/mpi_eigen_helpers.h"
#include "common/utils.h"

Eigen::VectorXd DeBoorParallelA(const ESMatrix &A, const ESMatrix &B, const Eigen::VectorXd &u) {
  using namespace Eigen;
  using namespace std;
  MPI_SETUP;
  MPI_Barrier(MPI_COMM_WORLD);

  unsigned long n = A.rows();
  unsigned long m = B.rows();
  assert(A.rows() == A.cols());
  assert(B.rows() == B.cols());
  if (!IsPowerOfTwo(n) || !IsPowerOfTwo(m)) {
    throw runtime_error(Format("The dimensions of the system being solved in parallel must be powers of two for "
                               "simplicity, but got n = %d and m = %d.", n, m));
  }

  int partition_rows = n / n_procs;
  int partition_cols = m / n_procs;

  // In alternative 1, we assume A, B, and G are available in each processor.
  // We factor them in each processor too, in order to prepare for the next step.
  SparseLU<SparseMatrix<double>> A_solver;
  A_solver.compute(A);
  SparseLU<SparseMatrix<double>> B_solver;
  B_solver.compute(B);
  MASTER {
      cout << "Deboor partial solvers done." << endl;
  };

  // TODO better name for this n x m matrix which is the resized RHS.
  MatrixXd G(u);
  cout << "Will reshape g (" << G.rows() << ", " << G.cols() << ") as a matrix: " << n << " x " << m << endl;
  G.resize(n, m);

  // This loop can be performed in parallel.
  MatrixXd local_D = MatrixXd::Zero(partition_rows, m);
  int local_start = local_id * partition_rows;
  int local_end = (local_id + 1) * partition_rows;
  for (int i = local_start; i < local_end; ++i) {
    // g_i is is the ith row in the g matrix.
    VectorXd g_i = G.block(i, 0, 1, m).transpose();
    local_D.row(i - local_start) = B_solver.solve(g_i).transpose();
  }
  cout << "Done first serial solver loop." << endl;

  // All-to-all transpose of the dense matrix D.
  MatrixXd local_D_transposed;
  cout << "Doing all to all!" << endl;
//  AllToAllEigenDense(local_D, local_D_transposed);

  // Return zeros for debugging for now.
  MatrixXd C = MatrixXd::Zero(n, m);
//  for (int j = 0; j < m; ++j) {
//    VectorXd d_prime_i = D.col(j);
//    C.col(j) = A_solver.solve(d_prime_i);
//  }
  C.resize(n * m, 1);
  return C;
}


#endif //HPSC_DEBOOR_PARALLEL_H
