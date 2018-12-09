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

using ESVector = Eigen::SparseVector<double>;


void GetARowUpper(int i, int n, int m, ESVector &rv) {
  // Procedurally generates a row from A.
  using namespace Eigen;
  rv.setZero();

  int row = i / m;
  int col = i % m;

  double factor = 1.0;
  if (row != 0 && row != n - 1) {
    factor *= 6.0;
  }
  if (col != 0 && col != m - 1) {
    factor *= 6.0;
  }
  rv.coeffRef(i) = factor;
  if (col != m - 1) {
    rv.coeffRef(i + 1) = 1;
    if (row != 0 && row != n - 1) {
      rv.coeffRef(i + 1) = 6.0;
    }
  }

  if (col != 0) {
    if (i + m - 2 + 1 < n * m) {
      rv.coeffRef(i + m - 2 + 1) = 1;
    }
  }
  if (i + m + 2 - 2 < n * m) {
    double fac = 1.0;
    if (col != 0 && col != m - 1) {
      fac = 6.0;
    }
    rv.coeffRef(i + m + 2 - 2) = fac;
  }
  if(col != m - 1) {
    if ( i + m + 3 - 2 < n * m) {
      rv.coeffRef(i + m + 3 - 2) = 1.0;
    }
  }
}


Eigen::SparseVector<double> GetARowLower(int i, int n, int m) {
  // Procedurally generates a lower half of a row from A.
  using namespace Eigen;
  Eigen::SparseVector<double> rv(n * m);

  int row = i / m;
  int col = i % m;

  double factor = 1.0;
  if (row != 0 && row != n - 1) {
    factor *= 6.0;
  }
  if (col != 0 && col != m - 1) {
    factor *= 6.0;
  }
  rv.insert(i) = factor;
  if (col != 0) {
    if (row != 0 && row != n - 1) {
      rv.insert(i - 1) = 6.0;
    }
    else {
      rv.insert(i - 1) = 1;
    }
  }

  if (col != 0) {
    if ( i - m - 3 + 2 >= 0) {
      rv.insert(i - m - 3 + 2) = 1.0;
    }
  }
  if (i - m - 2 + 2 >= 0) {
    double fac = 1.0;
    if (col != 0 && col != m - 1) {
      fac = 6.0;
    }
    rv.insert(i - m - 2 + 2) = fac;
  }
  if(col != m - 1) {
    if (i - m + 2 - 1 >= 0) {
      rv.insert(i - m + 2 - 1) = 1;
    }
  }

//  cout << "Returning row: " << i << endl;
//  cout << rv << endl;
  return rv;
}


void Computeq1(
    const std::vector<int> &row_indices,
    const ESMatrix &_,          // used to A, and only for debugging, now unused
    const Eigen::VectorXd &b,
               int n,
               int m,
               double w,
               const std::shared_ptr<Eigen::VectorXd> &x,
               Eigen::VectorXd &q1
) {
  Eigen::SparseVector<double> a_row(n * m);
  for(int i : row_indices) {
    int row = i / m;
    int col = i % m;
    GetARowUpper(i, n, m, a_row);
    a_row.coeffRef(i) = a_row.coeff(i) * (w - 1) / w;
    double val = a_row.dot((*x));

//      Eigen::MatrixXd res = (a_row * (*x));
//      cout << res.rows() << ", " << res.cols() << endl;
//      assert(res.rows() == 1 && res.cols() == 1);
//      double val = res(0, 0);

//      cout << val << ", " << val_hacky << endl;

    q1(i) = b(i) - val;
//    q1(i) = b(i) - val;
  }
}


/// Solves (L + D/w) * x = q1 for x, writing the result into x.
void ForwardSubst(
    const std::vector<int> &row_indices,
    const ESMatrix &_,    // ONLY for debugging
    int n,
    int m,
    double w,
    std::shared_ptr<Eigen::VectorXd> x,
    const Eigen::VectorXd &q1
)
{
  using namespace Eigen;
//  EMatrix A_built = EMatrix::Zero(A.rows(), A.cols());
  // TODO(andreib): Allow user to specify a coloring-based order, instead of just 1 .. n * m.

//  for (int i = 0; i < n * m; ++i) {
  for (int i : row_indices) {
    int row = i / m;
    int col = i % m;
    SparseVector<double> a_row = GetARowLower(i, n, m);
    double diag = a_row.coeff(i);

    double sum = 0.0;
    for (SparseVector<double>::InnerIterator it(a_row); it; ++it) {
      sum += it.value() * (*x)(it.row());
    }
    // Undo the last thing in loop; we have this outside the loop to avoid branching in the loop.
    sum = sum - diag * (*x)(i);

//    double sum_slow = 0.0;
//    for(int j = 0; j < i; ++j) {
//      sum_slow += a_row.coeffRef(j) * (*x)(j, 0);
//    }
//    if (fabs(sum - sum_slow) > 1e-3) {
//      cout << sum << "  " << sum_slow << endl;
//      cout << diag << " " << diag * (*x)(i, 0);
//    }

    double solution = (q1(i) - sum) / diag * w;
//    double solution = (q1(i) - sum_slow) / diag * w;
    (*x)(i) = solution;

//    A_built.row(i) = a_row;
  }

  // TODO(andreib): Remove this code.
//  if (A.rows() < 50) {
//    cout << "Demo A rebuilt:" << endl;
//    cout << A_built << endl;
//  }
}


std::shared_ptr<EMatrix> SOR(const ESMatrix &Ao, const Eigen::VectorXd &bo, int n, int m) {
  using namespace Eigen;
  using namespace std;

  // Hacky copies
  Eigen::VectorXd b(bo);
  ESMatrix A(Ao);

  const int kMaxIt = 100;
  // TODO pass as parameter
  const double w = 0.8;
  const bool reorder = false;

  vector<int> order;
  if (reorder) {
    cout << "REORDERING equations using 4-color coloring." << endl;
    // TODO simply iterate through q1 computation and forward subst using the coloring-based ordering.
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
    for(int i = 0; i < n * m; ++i) {
      order.push_back(i);
    }
  }

  if (A.rows() < 100) {
    cout << A << endl;
  }


  // Note: the natural ordering is what we've been doing so far in the previous cases.
  // Note: we need FOUR-COLOR coloring, not red-black. Two colors are not enough. And since we have >2 colors, we need
  // to decide which choice of color assignment to use. We should use the third one.
  auto x = make_shared<VectorXd>(VectorXd::Zero(n * m));

  ESMatrix L(A.triangularView<StrictlyLower>());
  ESMatrix U(A.triangularView<StrictlyUpper>());
  ESMatrix D(A.diagonal().asDiagonal());   // Extract diagonal as vector, and then turn

  // Note we just use A at the beginning (trivial to remove, but I chose to focus on optimizing the main loop, since
  // that's by far where the algorithms spends most of its time).
  VectorXd q0 = (L + D / w) * (*x);
//  VectorXd q1 = b - (U + (w - 1) / w * D) * (*x);

  VectorXd q1 = VectorXd::Zero(b.rows());
  Computeq1(order, A, b, n, m, w, x, q1);

  VectorXd r = q1 - q0;

  double kErrNormEps = 1e-12;
  double err_norm_0 = r.norm();
  double err_norm = r.norm();
  int iteration = 0;
  assert(order.size() == n * m);
  for(; iteration < kMaxIt; ++iteration) {
    if (err_norm < kErrNormEps * err_norm_0) {
      iteration--;
      break;
    }

//    ESMatrix M = (L + D / w);
//    *x = M.triangularView<Lower>().solve(q1);

    ForwardSubst(order, A, n, m, w, x, q1);
    cout << "FWD done" << endl;

    q0 = q1;
    // "Cheating" way
    //  q1 = b - (U + (w - 1) / w * D) * (*x);
    // Proper way, where 'A" is only passed for debugging.
    Computeq1(order, A, b, n, m, w, x, q1);
    cout << "q1 done" << endl;

    r = q1 - q0;
    err_norm = r.norm();
    if (iteration && iteration % 10 == 0) {
      cout << "[SOR] Iteration " << iteration << " complete. error = " << err_norm << endl;
    }
  }

  cout << "[SOR] Done in " << iteration << " its." << endl;

  auto x_mat = make_shared<EMatrix>(*x);
  x_mat->resize(n, m);
  return x_mat;
}


/*
 *
 * Code graveyard.

    double factor = 1.0;
    if (row != 0 && row != n - 1) {
      factor *= 6.0;
    }
    if (col != 0 && col != m - 1) {
      factor *= 6.0;
    }

    double val = factor * (w - 1) / w * (*x)(i);
    cout << i << ": fact = " << factor << ";";

    // Right neighbor, included everywhere it exists.
    if (col != m - 1) {
      double f2 = -1.0;
      if (row == 0 || row == n - 1) {
        f2 = 1.0;
      }
      else {
        f2 = 6.0;
      }
      val += f2 * (*x)(i + 1);

      cout << " Right neighbor f2 = " << f2 << ";";
    }

    // m + 1 neighbor, included everywhere except start of block
    if (col != 0) {
      if (i + m + 1 < n * m) {
        val += 1.0 * (*x)(i + m + 1);
        cout << " m+1 n'bor ;";
      }
    }

    // m + 2 neighbor, included everywhere
    if (i + m + 2 < n * m) {
      double f3 = 1.0;
      if (col != 0 && col != m - 1) {
        f3 = 6.0;
      }
      val += f3 * (*x)(i + m + 2);
      cout << " m+2 n'bor ;";
    }

    // m + 3 neighbor, included everywhere except end of block
    if (col != m - 1) {
      if (i + m + 3 < n * m) {
        val += 1.0 * (*x)(i + m + 3);
        cout << " m+3 n'bor ;";
      }
    }

    cout << endl;
    q1(i) = b(i) - val;
    continue;

    if (col == 0 || col == m - 1) {
      if (row == 0 || row == n - 1) {
        // Edge equations of first and last block
          val = (w - 1) / w * 1 * (*x)(i);

          if (i + 1 < n * m) {
            val += (*x)(i + 1) * 1;
          }
//          if (i + m + 1 < n * m) {
//            val += (*x)(i + m + 1);
//          }
          if (i + m + 2 < n * m) {
            val += (*x)(i + m + 2);
          }
          if (i + m + 3 < n * m) {
            val += (*x)(i + m + 3);
          }
      }
      else {
        // Edge equations of all but first and last block
        val = (w - 1) / w * 6 * (*x)(i);
        if (col == 0) {
          val += 6 * (*x)(i + 1);
        }
        if (i + m + 2 < n * m) {
          val += 1 * (*x)(i + m + 2);
        }
        if (i + m + 3 < n * m) {
          val += 1 * (*x)(i + m + 3);
        }
      }
    }
    else {
      if (row == 0 || row == n - 1) {
        // Non-edge equations of first and last block
        val = ((w - 1) / w * 6 * (*x)(i));
        if (i + 1 < n * m) {
          val += (*x)(i + 1);
        }
        if (i + m + 1< n * m) {
          val += 1 * (*x)(i + m + 1);
        }
        if (i + m + 2 < n * m) {
          val += 6 * (*x)(i + m + 2);
        }
        if (i + m + 3 < n * m) {
          val += 1 * (*x)(i + m + 3);
        }
      }
      else {
        // Non-edge equation of all but first and last block (general case)
        val = ((w - 1) / w * 36 * (*x)(i));
        if (i + 1 < n * m) {
          val += 6 * (*x)(i + 1);
        }
        if (i + m + 1 < n*m) {
          val += (*x)(i + m + 1);
        }
        if (i + m + 2 < n*m) {
          val += 6 * (*x)(i + m + 2);
        }
        if (i + m + 3 < n * m) {
          val += (*x)(i + m + 3);
        }
      }
    }

    q1(i) = b(i) - val;
  }

//  if (A_built.rows() < 100) {
//    cout << "The row-wise built A:" << endl;
//    cout << A_built << endl;
//  }
 */

#endif //HPSC_SOR_H
