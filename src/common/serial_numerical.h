//
// Serial numerical routines, often used as building blocks for more complex parallel ones.
//

#ifndef HPSC_SERIAL_NUMERICAL_H
#define HPSC_SERIAL_NUMERICAL_H

#include <cassert>
#include <vector>
#include "matrix.h"

// Bad practice, but OK for small projects like this one.
using namespace std;

/// Performs LU factorization of the banded matrix A, in-place.
void BandedLUFactorization(BandMatrix<double> &A, bool check_lu);

/// Solves the system given A, a matrix assumed to have already been LU-decomposed in-place.
Matrix<double> SolveDecomposed(const BandMatrix<double> &A_decomposed, Matrix<double> &B);


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
Matrix<double> SolveSerial(BandMatrix<double> &A, Matrix<double> &B, bool check_lu = false);


/// Polynomial interpolation basis function.
template<typename T>
T Phi(T x) {
  if (x >= 0 && x <= 1) {
    return 0.5 * x * x;
  }
  else if(x > 1 && x <= 2) {
    return 0.5 * (-2.0 * (x - 1) * (x - 1) + 2 * (x - 1) + 1);
  }
  else if(x > 2 && x <= 3) {
    return 0.5 * (3 - x) * (3 - x);
  }
  else {
    return 0.0;
  }
}


template<typename T>
T PhiI(uint32_t i, T x, T a, uint32_t n, T step_size) {
  assert(i >= 0 && i <= n + 1);
  return Phi((x - a) / step_size - i + 2);
}


#endif //HPSC_SERIAL_NUMERICAL_H
