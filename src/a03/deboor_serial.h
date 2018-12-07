/**
 *  @file deboor_serial.h
 *  @brief Logic for solving linear systems of the form $(A \otimes B) x = g$ for $x$.
 */

#ifndef HPSC_DEBOOR_H
#define HPSC_DEBOOR_H

#include <Eigen/Core>

#include "common/eigen_helpers.h"
#include "common/mpi_helpers.h"
#include "common/mpi_stopwatch.h"
#include "a03/deboor_common.h"

/// Solves a linear system defined as KroneckerProduct(A, B) x = u serially using DeBoor decomposition.
/// \param A First component of the Kronecker product, [n x n].
/// \param B Second component of the Kronecker product, [m x m].
/// \param u Right-had side column vector, [nm x 1].
/// \return The [nm x 1] solution vector.
Eigen::VectorXd DeBoorDecomposition(const ESMatrix &A,
                                    const ESMatrix &B,
                                    const Eigen::VectorXd &u,
                                    MPIStopwatch &stopwatch) {
  using namespace Eigen;
  using namespace std;
  MPI_SETUP;
  assert(A.rows() == A.cols());
  assert(B.rows() == B.cols());
  int n = A.rows();
  int m = B.rows();
  stopwatch.Record("init");

  SparseLU<SparseMatrix<double>> A_solver;
  A_solver.compute(A);
  SparseLU<SparseMatrix<double>> B_solver;
  B_solver.compute(B);
  MASTER {
    cout << "Deboor partial solvers done." << endl;
  };
  stopwatch.Record("factorization");

  // TODO better name for this n x m matrix which is the resized RHS.
  MatrixXd G(u);
  cout << "Will reshape g as a matrix: " << n << " x " << m << endl;
  G.resize(n, m);

  // This loop can be performed in parallel.
  MatrixXd D = MatrixXd::Zero(n, m);
  for (int i = 0; i < n; ++i) {
    // g_i is is the ith row in the g matrix.
    VectorXd g_i = G.block(i, 0, 1, m).transpose();
//    VectorXd g_i = G.block(i, 0, 1, m).transpose();
//    cout << g_i.rows() << " x " << g_i.cols() << endl;
// TODO remove profanity
// WHY THE FUCK DOES THIS WORK WITH AND WITHOUT TRANSPOSE BUT PRODUCE DIFFERENT RESULTS?
    D.row(i) = B_solver.solve(g_i).transpose();
  }
//  cout << "Done first serial solver loop." << endl;
  stopwatch.Record("first_stage");
  stopwatch.Record("transpose_stage");

  // This will be a communication bottleneck, since we need to transpose D here.
  // In the serial version we never have to explicitly transpose, since we can just change our access pattern to
  // pretend we transposed the matrix.
  // Now D's rows are d'_i, not d_i.

  MatrixXd C = MatrixXd::Zero(n, m);
  for (int j = 0; j < m; ++j) {
    VectorXd d_prime_i = D.col(j);
    C.col(j) = A_solver.solve(d_prime_i);
  }
  C.resize(n * m, 1);
  stopwatch.Record("second_stage");
  return C;
}



#endif //HPSC_DEBOOR_H
