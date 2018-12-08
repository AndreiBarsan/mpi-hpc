//
// Created by andreib on 12/7/18.
//

#ifndef HPSC_SOR_H
#define HPSC_SOR_H

#include <memory>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "common/matrix.h"
#include "common/mpi_eigen_helpers.h"
#include "common/utils.h"

// TODO(andreib): Do NOT pass A explicitly; just use the rule described in handout.
std::shared_ptr<Eigen::MatrixXd> SOR(const ESMatrix &A, const Eigen::VectorXd &b, int n, int m) {
  using namespace Eigen;
  using namespace std;

  const int kMaxIt = 100;
  // TODO pass as parameter
  const double w = 0.8;

  // Note: the natural ordering is what we've been doing so far.
  auto x = make_shared<MatrixXd>(MatrixXd::Zero(n * m, 1));

  ESMatrix L(A.triangularView<StrictlyLower>());
  ESMatrix U(A.triangularView<StrictlyUpper>());
  ESMatrix D(A.diagonal().asDiagonal());   // Extract diagonal as vector, and then turn
  // it into a matrix.
//  cout << D.rows() << ", " << D.cols() << endl;
//  cout << D.asDiagonal().rows() << ", " << D.asDiagonal().cols() << endl;

  VectorXd q0 = (L + D / w) * (*x);
  VectorXd q1 = b - (U + (w - 1) / w * D) * (*x);
  VectorXd r = q1 - q0;

  double kErrNormEps = 1e-12;
  double err_norm_0 = r.norm();
  double err_norm = r.norm();
  for(int iteration = 0; iteration < kMaxIt; ++iteration) {
    if (err_norm < kErrNormEps * err_norm_0) {
      iteration--;
      break;
    }

    ESMatrix M = (L + D / w);
    *x = M.triangularView<Lower>().solve(q1);
    q0 = q1;
    q1 = b - (U + (w - 1) / w * D) * (*x);
    r = q1 - q0;
    err_norm = r.norm();
    if (iteration && iteration % 10 == 0) {
      cout << "[SOR] Iteration " << iteration << " complete. error = " << err_norm << endl;
    }
  }

  x->resize(n, m);
  return x;
}

#endif //HPSC_SOR_H
