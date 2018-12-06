/**
 * @file deboor_parallel.h
 * @brief Functions for solving linear systems in parallel using DeBoor decomposition.
 */

#ifndef HPSC_DEBOOR_PARALLEL_H
#define HPSC_DEBOOR_PARALLEL_H

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "common/matrix.h"
#include "common/mpi_eigen_helpers.h"
#include "common/parallel_numerical.h"
#include "common/utils.h"

Eigen::VectorXd DeBoorParallelA(const ESMatrix &A, const ESMatrix &B, const Eigen::VectorXd &u) {
  using namespace Eigen;
  using namespace std;
  MPI_SETUP;
  if (n_procs <= 1) {
    throw runtime_error("Please run this with 'mpirun' on at least 2 processors (4 recommended).");
  }
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

  // In alternative 1, we assume A, B, and rhs_matrix are available in each processor.
  // We factor them in each processor too, in order to prepare for the next step.
  SparseLU<SparseMatrix<double>> A_solver;
  A_solver.compute(A);
  SparseLU<SparseMatrix<double>> B_solver;
  B_solver.compute(B);
  MASTER {
      cout << "Deboor partial solvers done." << endl;
  };

  // Represents the RHS of the linear system as a matrix, instead of a vector.
  MatrixXd rhs_matrix(u);
  rhs_matrix.resize(n, m);

  MatrixXd local_D = MatrixXd::Zero(partition_rows, m);
  int local_start = local_id * partition_rows;
  int local_end = (local_id + 1) * partition_rows;
  for (int i = local_start; i < local_end; ++i) {
    VectorXd g_i = rhs_matrix.row(i);
    local_D.row(i - local_start) = B_solver.solve(g_i).transpose();
  }
  cout << "Done first parallel solver loop." << endl;

  // All-to-all transpose of the dense matrix D.
  MatrixXd local_D_transposed;
  TransposeEigenDense(local_D, local_D_transposed);

  MatrixXd C = MatrixXd::Zero(n, partition_cols);
  local_start = local_id * partition_cols;
  local_end = (local_id + 1) * partition_cols;
  for(int i = local_start; i < local_end; ++i) {
    VectorXd d_prime_i = local_D_transposed.row(i - local_start);
    C.col(i - local_start) = A_solver.solve(d_prime_i);
  }
  cout << "Done second parallel solver loop." << endl;

  VectorXd C_full = VectorXd::Zero(n * m);
  MPI_Allgather(
      C.data(),
      n * partition_cols,
      MPI_DOUBLE,
      C_full.data(),
      n * partition_cols,
      MPI_DOUBLE,
      MPI_COMM_WORLD);
  return C_full;
}


Eigen::VectorXd DeBoorParallelB(const ESMatrix &A, const ESMatrix &B, const Eigen::VectorXd &u) {
  // TODO(andreib): Don't assume each node knows A!
  // TODO(andreib): Eliminate using statements.
  using namespace Eigen;
  using namespace std;
  MPI_SETUP;
  if (n_procs <= 1) {
    throw runtime_error("Please run this with 'mpirun' on at least 2 processors (4 recommended).");
  }
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

  // In alternative 2, we assume B, and rhs_matrix are available in each processor, but A to be distributed among the
  // processors.
//  SparseLU<SparseMatrix<double>> A_solver; A_solver.compute(A);
  SparseLU<SparseMatrix<double>> B_solver; B_solver.compute(B);

  // The first stage of Alternative 2 (here, called "B") is the same as in the first one.
  MatrixXd rhs_matrix(u);
  rhs_matrix.resize(n, m);
  MatrixXd local_D = MatrixXd::Zero(partition_rows, m);
  int local_start = local_id * partition_rows;
  int local_end = (local_id + 1) * partition_rows;
  for (int i = local_start; i < local_end; ++i) {
    VectorXd g_i = rhs_matrix.row(i);
    local_D.row(i - local_start) = B_solver.solve(g_i).transpose();
  }
  cout << "Done first parallel solver loop." << endl;

  // TODO(andreib): SolveParallel implementation of PP2 assumes the full A is on master and splits it up to everyone.
  // You should update the implementation to account for the fact that A is already distributed row-wise in this
  // problem.
  ::BandMatrix<double> A_custom = ToTridiagonalMatrix(A);
  ::Matrix<double> local_D_custom = ToMatrix(local_D);
  ::Matrix<double> distributed_solution = ::SolveParallel(A_custom, local_D_custom, true);

  EMatrix distributed_solution_eigen = ToEigen(distributed_solution);
  distributed_solution_eigen.resize(n * m, 1);
  return distributed_solution_eigen;
}


#endif //HPSC_DEBOOR_PARALLEL_H
