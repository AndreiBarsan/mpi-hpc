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

std::shared_ptr<Eigen::MatrixXd> SOR(const ESMatrix &A, const Eigen::VectorXd &b, int n, int m) {
  using namespace Eigen;
  using namespace std;

  const int kMaxIt = 100;
  // TODO pass as parameter
  const double w = 0.8;
  const bool reorder = true;

  if (reorder) {
    cout << "REORDERING equations using 4-color coloring." << endl;
    // TODO permute A and b using a permutation matrix P.
    //
    //  R B R     <=> i even
    //  G Y G     <=> i odd
    //  R B R     <=> i even
    //  ...etc.
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        int idx = i * m + j;

        if (i % 2 == 0 && j % 2 == 0) {
          // RED
        }
        else if(i % 2 == 0 && j % 2 == 1) {
          // BLUE
        }
        else if (i % 2 == 1 && j % 2 == 0) {
          // GREEN
        }
        else {
          // YELLOW
        }
      }

    }
  }
  else {
    cout << "NOT reordering equations using coloring." << endl;
  }

  // Note: the natural ordering is what we've been doing so far in the previous cases.
  // Note: we need FOUR-COLOR coloring, not red-black. Two colors are not enough. And since we have >2 colors, we need
  // to decide which choice of color assignment to use. We should use the third one.
  auto x = make_shared<MatrixXd>(MatrixXd::Zero(n * m, 1));

  ESMatrix L(A.triangularView<StrictlyLower>());
  ESMatrix U(A.triangularView<StrictlyUpper>());
  ESMatrix D(A.diagonal().asDiagonal());   // Extract diagonal as vector, and then turn

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

/* Manual computation code.
 *
    // I do not understand why we have to do this. Storing A itself can be done efficiently enough to allow all of
    // these insane for loops to be replaced by one line of code, whose performance WILL be faster than the manual
    // loops thanks to Eigen's auto-vectorization capabilities.
//    q1(0) = ...
    for (int i = 1; i < m - 1; ++i) {
      q1(i) = b(i) - (( (w - 1) / w * 6 * (*x)(i) + (*x)(i + 1) + (*x)(i + m + 1) ))
    }
//    q1(m - 1) = ...
    for (int i = 0; i < n; ++i) {
      for(int j = 0; j < m; ++j) {
        int idx = i * m + j;

        double val = 0.0;

        // First block, 6 1 ... 1 6 1, unless it's first or last row in first block
        if (i == 0) {
          if (j == 0 || j == m -1) {
            val += (w - 1) / w * 1 * (*x)
          }


        }

        q1(idx) = val;
      }

 //*/

/**
//    for (int i = m; i < (n - 1) * m; ++i) {
//      // 1 6 1 ... 6 36 6 ... 1 6 1   => but we care only about the U, so
//      //             36 6 ... 1 6 1
//      q1(i) = b(i) - ((w - 1) / w * 36 * (*x)(i) + 6 * (*x)(i + 1) + (*x)(i + m + 1) + 6 * (*x)(i + m + 2) + (*x)(i + m + 3) );
//    }
 */

#endif //HPSC_SOR_H
