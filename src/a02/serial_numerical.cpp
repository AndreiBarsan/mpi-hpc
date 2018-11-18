#include "serial_numerical.h"

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
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < n; ++j) {
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

Matrix<double> SolveSerial(BandMatrix<double> &A, Matrix<double> &B, bool check_lu) {
  assert(B.rows_ == A.get_n());
  cout << "Will solve " << B.cols_ << " linear systems." << endl;
  // Start by factorizing A in-place.
  BandedLUFactorization(A, check_lu);
  return SolveDecomposed(A, B);
}
