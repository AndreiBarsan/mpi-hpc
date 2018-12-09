/**
 * @file sor.h
 * @brief Serial code for solving 2D spline interpolation using SOR. (Not a generic solver.)
 */

#ifndef HPSC_SOR_H
#define HPSC_SOR_H

#include <memory>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "common/matrix.h"
#include "common/mpi_eigen_helpers.h"
#include "common/utils.h"

using ESVector = Eigen::SparseVector<double>;


void GenerateColoredOrder(int n, int m, vector<int> &order);


void GenerateNaturalOrder(int n, int m, vector<int> &order) {
  for (int i = 0; i < n * m; ++i) {
    order.push_back(i);
  }
}


void GetARowUpper(int i, int n, int m, ESVector &rv) {
  // Procedurally generates the upper half of a row from A.
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


void Computeq0(
    const std::vector<int> &row_indices,
    int n,
    int m,
    double w,
    const std::shared_ptr<Eigen::VectorXd> &x,
    Eigen::VectorXd &q0
) {
//  Eigen::SparseVector<double> a_row(n * m);
  for(int i : row_indices) {
    int row = i / m;
    int col = i % m;
    Eigen::SparseVector<double> a_row = GetARowLower(i, n, m);
    a_row.coeffRef(i) = a_row.coeff(i) / w;
    double val = a_row.dot((*x));

//      Eigen::MatrixXd res = (a_row * (*x));
//      cout << res.rows() << ", " << res.cols() << endl;
//      assert(res.rows() == 1 && res.cols() == 1);
//      double val = res(0, 0);

//      cout << val << ", " << val_hacky << endl;

    q0(i) = val;
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
  }
}

/// Solves a linear system arising from the 2D quadratic spline interpolation problem using SOR.
/// \warning Not a generic linear solver.
std::shared_ptr<EMatrix> SOR(const ESMatrix &_,
                             const Eigen::VectorXd &b,
                             int n,
                             int m,
                             double w,
                             bool reorder,
                             int *out_iterations) {
  using namespace Eigen;
  using namespace std;
  const int kMaxIt = 100;
  vector<int> order;
  order.reserve(n * m);

  // Note: the natural ordering is what we've been doing so far in the previous cases.
  // Note: we need FOUR-COLOR coloring, not red-black. Two colors are not enough. And since we have >2 colors, we need
  // to decide which choice of color assignment to use. We should use the third one.
  if (reorder) {
//    cout << "REORDERING equations using 4-color coloring." << endl;
    GenerateColoredOrder(n, m, order);
  }
  else {
//    cout << "NOT reordering equations using coloring." << endl;
    GenerateNaturalOrder(n, m, order);
  }

  // This is the solution we will be iterating on. It is initialized as all-zeros.
  auto x = make_shared<VectorXd>(VectorXd::Zero(n * m));

//  ESMatrix L(A.triangularView<StrictlyLower>());
//  ESMatrix U(A.triangularView<StrictlyUpper>());
//  ESMatrix D(A.diagonal().asDiagonal());   // Extract diagonal as vector, and then turn

//  VectorXd q0 = (L + D / w) * (*x);
//  VectorXd q1 = b - (U + (w - 1) / w * D) * (*x);

  // Algorithm initialization logic, WITHOUT using x.
  VectorXd q0(VectorXd::Zero(b.rows()));
  Computeq0(order, n, m, w, x, q0);

  VectorXd q1 = VectorXd::Zero(b.rows());
  Computeq1(order, _, b, n, m, w, x, q1);
  VectorXd r = q1 - q0;

  // Use 10^{-9} as the convergence tolerance, as instructed in the handout.
  double kErrNormEps = 1e-9;
  double err_norm_0 = r.norm();
  double err_norm = r.norm();
  int iteration = 0;

  cout << "Initialized computation equation ordering." << endl;
  assert(order.size() == n * m);
  for(; iteration < kMaxIt; ++iteration) {
    if (err_norm < kErrNormEps * err_norm_0) {
      iteration--;
      break;
    }

    // Note that I only pass A to these equations for historic reasons (debugging). A is NEVER actually used.
    ForwardSubst(order, _, n, m, w, x, q1);
    q0 = q1;
    Computeq1(order, _, b, n, m, w, x, q1);

    r = q1 - q0;
    err_norm = r.norm();
//    if (iteration && iteration % 10 == 0) {
//      cout << "[SOR] Iteration " << iteration << " complete. error = " << err_norm << endl;
//    }
  }
//  cout << "[SOR] Done in " << iteration << " its." << endl;

  *out_iterations = iteration;
  auto x_mat = make_shared<EMatrix>(*x);
  x_mat->resize(n, m);
  return x_mat;
}

void GenerateColoredOrder(int n, int m, vector<int> &order) {
  //  R B R     <=> i even
  //  G Y G     <=> i odd
  //  R B R     <=> i even
  //  ...etc.
  vector<int> red_idx;
  red_idx.reserve(n * m / 3);
  vector<int> blue_idx;
  blue_idx.reserve(n * m / 3);
  vector<int> green_idx;
  green_idx.reserve(n * m / 3);
  vector<int> yellow_idx;
  yellow_idx.reserve(n * m / 3);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < m; ++col) {
      int i = row * m + col;
      // Coloring 3
      if (row % 2 == 0) {
        if (col % 2 == 0) {
          red_idx.push_back(i);
        } else {
          blue_idx.push_back(i);
        }
      } else {
        if (col % 2 == 0) {
          green_idx.push_back(i);
        } else {
          yellow_idx.push_back(i);
        }
      }
      continue;
      // Coloring 2
//        switch(row % 4) {
//          case 0:
//            if (col % 2 == 0) {
//              red_idx.push_back(i);
//            }
//            else {
//              green_idx.push_back(i);
//            }
//            break;
//          case 1:
//            if (col % 2 == 0) {
//              yellow_idx.push_back(i);
//            }
//            else {
//              blue_idx.push_back(i);
//            }
//            break;
//          case 2:
//            if (col % 2 == 0) {
//              green_idx.push_back(i);
//            }
//            else {
//              red_idx.push_back(i);
//            }
//            break;
//          case 3:
//            if (col % 2 == 0) {
//              blue_idx.push_back(i);
//            }
//            else {
//              yellow_idx.push_back(i);
//            }
//            break;
//          default:
//            throw runtime_error("Math stopped working.");
//        }
    }
  }
  order.insert(order.end(), red_idx.cbegin(), red_idx.cend());
  order.insert(order.end(), blue_idx.cbegin(), blue_idx.cend());
  order.insert(order.end(), green_idx.cbegin(), green_idx.cend());
  order.insert(order.end(), yellow_idx.cbegin(), yellow_idx.cend());
}


/*
 *
 * Code graveyard.
 */
/*
 * Generic SOR loop
 *
    ESMatrix M = (L + D / w);
    *x = M.triangularView<Lower>().solve(q1);
    q1 = b - (U + (w - 1) / w * D) * (*x);
 */


/*

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
