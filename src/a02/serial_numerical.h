//
// Serial numerical routines, often used as building blocks for more complex parallel ones.
//

#ifndef HPSC_SERIAL_NUMERICAL_H
#define HPSC_SERIAL_NUMERICAL_H

#include <cassert>
#include <vector>

#ifdef DEBUG_WITH_EIGEN
#include "Eigen/Eigen"
#endif


#include "matrix.h"

// Bad practice, but OK for small projects like this one.
using namespace std;

/// Performs LU factorization of the banded matrix A, in-place.
void BandedLUFactorization(BandMatrix<double> &A, bool check_lu) {
  // For debugging
  Matrix<double> A_orig = A.get_dense();
  unsigned int n = A.get_n();

  unsigned int l = A.get_bandwidth();
  unsigned int u = A.get_bandwidth();

  for (unsigned int k = 0; k < n - 1; ++k) {
    for (unsigned int i = k + 1; i <= min(k + l, n - 1); ++i) {
      A(i, k) = A(i, k) / A(k, k);

      for (unsigned int j = k + 1; j <= min(k + u, n - 1); ++j) {
        A(i, j) = A(i, j) - A(i, k) * A(k, j);
      }
    }
  }

  if (check_lu) {
    vector<double> lower_data;
    vector<double> upper_data;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i > j) {
          lower_data.push_back(A.get(i, j));
          upper_data.push_back(0.0);
        } else {
          lower_data.push_back(0.0);
          upper_data.push_back(A.get(i, j));
        }
      }
    }

    Matrix<double> lower(n, n, lower_data);
    // Ensure we have the implicit 1's on the diagonal.
    for (uint32_t i = 0; i < n; ++i) {
      lower(i, i) = 1.0;
    }
    Matrix<double> upper(n, n, upper_data);
    bool all_close = A_orig.all_close(lower * upper);
    if (!all_close) {
      throw runtime_error("LU-decomposition is incorrect! Get a refund! ;)");
    }
  }
}

/// Solves the system given A, a matrix assumed to have already been LU-decomposed in-place.
Matrix<double> SolveDecomposed(const BandMatrix<double> &A_decomposed, Matrix<double> &B) {
  int n_systems = B.cols_;
  unsigned int n = A_decomposed.get_n();

  // Perform forward substitution to find intermediate result z.
  Matrix<double> z(B);
  int bw = A_decomposed.get_bandwidth();
  for (int i = 0; i < n; ++i) {
    for(int j = max(0, i - bw); j <= i - 1; ++j) {
      // Solve all linear systems at the same time.
      for(int k = 0; k < n_systems; ++k) {
        B(i, k) = B(i, k) - A_decomposed.get(i, j) * z(j, k);
      }
    }

    for(int k = 0; k < n_systems; ++k) {
      z(i, k) = B(i, k); // / A(i, i);  // No divide because lower always has a 1 on the diagonal!
    }
  }

  // Perform back substitution
  // We store our output here and the rhs is z.
  Matrix<double> x(B);
  for (int j = n - 1; j >= 0; --j) {
    for(int k = 0; k < n_systems; ++k) {
      // the upper matrix has non-ones on diag, so we DO need to divide!
      x(j, k) = z(j, k) / A_decomposed.get(j, j);
      for (int i = max(0, j - bw); i <= max(0, j - 1); i++) {
        z(i, k) = z(i, k) - A_decomposed.get(i, j) * x(j, k);
      }
    }
  }
  return x;
}


/// Solves the given banded linear systems in-place using a LU factorization.
///
/// \param A [n x n] coefficient matrix.
/// \param B [n x k] right-hand side (solves k systems at the same time).
/// \param check_lu Whether to validate the result of the LU factorization.
/// \return An [n x k] matrix with the k n-dimensional solution vectors.
///
/// Note: Destroys A and b by performing the factorization and substitutions in-place.
/// High-level overview of the solver code:
///     1. Want to solve: AX = B
///     2. LU decompose A: A = LU
///     3. System is now: LUX = B
///     4. Let UX := Z and solve LZ = B for Z with forward substitution.
///     5. Solve UX = Z for X with backward substitution.
///     6. Return the final solutions X.
Matrix<double> SolveSerial(BandMatrix<double> &A, Matrix<double> &B, bool check_lu = false) {
  assert(B.rows_ == A.get_n());
  cout << "Will solve " << B.cols_ << " linear systems." << endl;
  // Start by factorizing A in-place.
  BandedLUFactorization(A, check_lu);
  return SolveDecomposed(A, B);
}

#include "src/a02/matrix.h"

#endif //HPSC_SERIAL_NUMERICAL_H
