//
// Very ugly but reasonably efficient matrix classes.
//

#ifndef HPSC_MATRIX_H
#define HPSC_MATRIX_H

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include "common/eigen_helpers.h"
#include "common/utils.h"

using EMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using namespace std;

/// A quick and dirty row-major dense matrix class.
template<typename T>
class Matrix {
 public:
    Matrix(uint32_t n, uint32_t m, const std::vector<T> &data)
        : rows_(n), cols_(m), data_(data)
    {
      assert (n * m == data.size());
    }

    T& operator()(int n, int m) {
      assert (n >= 0 && static_cast<uint32_t>(n) < rows_);
      assert (m >= 0 && static_cast<uint32_t>(m) < cols_);
      return data_[n * cols_ + m];
    }

    T operator()(int n, int m) const {
      assert (n >= 0 && static_cast<uint32_t>(n) < rows_);
      assert (m >= 0 && static_cast<uint32_t>(m) < cols_);
      return data_[n * cols_ + m];
    }

    bool all_close(const Matrix<T> &other, bool fail_on_nan = true) const {
      assert (rows_ == other.rows_ && cols_ == other.cols_);
      T epsilon = 1e-6;
      for(uint32_t i = 0; i < rows_; ++i) {
        for(uint32_t j = 0; j < cols_; ++j) {
          if (std::fabs((*this)(i, j) - other(i, j) > epsilon)) {
            return false;
          }

          if(fail_on_nan && (std::isnan((*this)(i, j)) || std::isnan(other(i, j)))) {
            return false;
          }
        }
      }
      return true;
    }

    uint32_t write_raw_rows(uint32_t row_start, uint32_t row_end, T *out, uint32_t offset) const {
      uint32_t idx = offset;
      for(uint32_t i = row_start; i < row_end; ++i) {
        for(uint32_t j = 0; j < cols_; ++j) {
          out[idx++] = data_[i * cols_ + j];
        }
      }
      return idx;
    }

    uint32_t write_raw(unique_ptr<T[]> &p, uint32_t offset) const {
      return write_raw(p.get(), offset);
    }

    uint32_t write_raw(T *out, uint32_t offset) const {
      return write_raw_rows(0, rows_, out, offset);
    }

    /// Sets the values of this matrix from the given buffer, in row-major order.
    /// Useful for reading data received via MPI.
    uint32_t set_from(unique_ptr<T[]> &p, uint32_t offset = 0) {
      return set_from(p.get(), offset);
    }

    uint32_t set_from(T *raw, uint32_t offset = 0) {
      uint32_t cur_offset = offset;
      for(uint32_t i = 0; i < rows_; ++i) {
        for(uint32_t j = 0; j < cols_; ++j) {
          uint32_t idx = i * cols_ + j;
          data_[idx] = raw[cur_offset++];
        }
      }
      return cur_offset;
    }

    /// Euclidean norm for vectors, Frobenius for matrices.
    double norm() const {
      double sum_sq = 0.0;
      for(uint32_t i = 0; i < rows_; ++i) {
        for(uint32_t j = 0; j < cols_; ++j) {
          sum_sq += (*this)(i, j) * (*this)(i, j);
        }
      }
      return sqrt(sum_sq);
    }

 public:
  uint32_t rows_, cols_;

 private:
  std::vector<T> data_;
};

/// Implements a square banded matrix. Stores data in compact row-major order. Assumes each row has 2 * band + 1
/// elements except the first and the last 'band'.
///
/// While the original problem for quadratic spline interpolation is just tridiagonal, the final reduced system is
/// banded with a wider band, so we do need to implement our matrices to support this.
template<typename T>
class BandMatrix {

 public:
  BandMatrix(uint32_t n, const std::vector<T> &data, uint32_t bandwidth=1UL)
      : n_(n),
        bandwidth_(bandwidth),
        data_(data)
  {
    // Check that we got the right number of elements.
    uint32_t effective_bw = 2 * bandwidth_ + 1;
//    uint32_t missing_at_edge = bandwidth_ * (bandwidth_ + 1) / 2;
    assert(data.size() == (effective_bw) * n); // - 2 * missing_at_edge);
  }

  T& operator()(int row_id, int col_id) {
    assert(row_id >= 0 && static_cast<uint32_t>(row_id) < n_);
    assert(col_id >= 0 && static_cast<uint32_t>(col_id) < n_);

    if (abs(col_id - row_id) <= bandwidth_) {
      return data_.at(index_(row_id, col_id));
    }
    else {
      throw std::out_of_range("Cannot access off-banded-diagonal element in non-const way.");
    }
  }

  // TODO(andreib): Make this more consistent.
  T get(int row_id, int col_id) const {
    assert(row_id >= 0 && static_cast<uint32_t>(row_id) < n_);
    assert(col_id >= 0 && static_cast<uint32_t>(col_id) < n_);

    if (abs(col_id - row_id) <= bandwidth_) {
      return data_[index_(row_id, col_id)];
    }
    else {
      return 0;
    }
  }

  BandMatrix& operator*(const T& other_scalar) {
    for (T &val : data_) {
      val *= other_scalar;
    }
    return *this;
  }

  uint32_t get_n() const {
    return n_;
  }

  /// Returns a dense representation of the data in this matrix.
  Matrix<T> get_dense() const {
    std::vector<T> result_data;
    for (uint32_t i = 0; i < n_; ++i) {
      for (uint32_t j = 0; j < n_; ++j) {
        result_data.push_back(this->get(i, j));
      }
    }
    return Matrix<T>(n_, n_, result_data);
  }

  uint32_t get_bandwidth() const {
    return bandwidth_;
  }

  /// Writes the raw data corresponding to the rows in [row_start, row_end) to out, returning the number of elements
  /// written. Note that this also writes padding elements.
  uint32_t write_raw_rows(uint32_t row_start, uint32_t row_end, T *out, uint32_t offset) const {
    uint32_t idx = offset;
    uint32_t effective = bandwidth_ * 2 + 1;
    for(uint32_t i = row_start; i < row_end; ++i) {
      for (uint32_t raw_col = 0; raw_col < effective; ++raw_col) {
        out[idx++] = data_[i * effective + raw_col];
      }
    }
    return idx;
  }

 private:
  uint32_t bandwidth_;
  uint32_t n_;
  std::vector<T> data_;

  int index_(int row_id, int col_id) const {
    int effective = bandwidth_ * 2 + 1;
    int off = col_id - row_id + bandwidth_;
    return row_id * effective + off;
  }
};

//bool all_close(const std::vector<double> &left, const std::vector<double> &right) {
//  double epsilon = 1e-6;
//  for(uint32_t i = 0; i < left.size(); ++i) {
//    if (fabs(left[i] - right[i]) > epsilon) {
//      return false;
//    }
//  }
//  return true;
//}

EMatrix ToEigen(const Matrix<double> &mat);;

Matrix<double> ToMatrix(const EMatrix &eigen);

/// Hacky method which assumes the given sparse eigen matrix is tridiagonal, but does NOT check that!
BandMatrix<double> ToTridiagonalMatrix(const ESMatrix &eigen);

EMatrix ToEigen(const BandMatrix<double> &mat);;



template<typename T>
Matrix<T> operator*(const Matrix<T> &left, const Matrix<T> &right) {
  assert(left.cols_ == right.rows_);
  std::vector<T> res_data;
  for(uint32_t i = 0; i < left.rows_ * right.cols_; ++i) {
    res_data.push_back(0.0);
  }

  Matrix<T> result(left.rows_, right.cols_, res_data);
  for(uint32_t i = 0; i < left.rows_; ++i) {
    for(uint32_t j = 0; j < right.cols_; ++j) {
      for(uint32_t k = 0; k < left.cols_; ++k) {
        result(i, j) += left(i, k) * right(k, j);
      }
    }
  }
  return result;
}

template<typename T>
Matrix<T> operator-(const Matrix<T> &left, const Matrix<T> &right) {
  assert(left.cols_ == right.cols_);
  assert(left.rows_ == right.rows_);

  Matrix<T> result(left);
  for(uint32_t i = 0; i < left.rows_; ++i) {
    for(uint32_t j = 0; j < left.cols_; ++j) {
        result(i, j) -= right(i, j);
    }
  }
  return result;
}

template<typename T>
Matrix<T> operator+(const Matrix<T> &left, const Matrix<T> &right) {
  assert(left.cols_ == right.cols_);
  assert(left.rows_ == right.rows_);

  Matrix<T> result(left);
  for(uint32_t i = 0; i < left.rows_; ++i) {
    for(uint32_t j = 0; j < left.cols_; ++j) {
      result(i, j) += right(i, j);
    }
  }
  return result;
}

/// Useful for printing a matrix as text to a stream. (Includes all zeros.)
template<typename T>
std::ostream& operator<<(std::ostream& out, const BandMatrix<T> &m) {
  uint32_t n = m.get_n();

  for(uint32_t i = 0; i < n; ++i) {
    for(uint32_t j = 0; j < n; ++j) {
      out << setw(6) << setprecision(4) << m.get(i, j) << " ";
    }
    out << "\n";
  }

  return out;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T> &m) {
  for(uint32_t i = 0; i < m.rows_; ++i) {
    for(uint32_t j = 0; j < m.cols_; ++j) {
      out << setw(6) << setprecision(4) <<  m(i, j);
      if (j != m.cols_ - 1 || m.cols_ == 1) {
        out << ", ";
      }
    }
    // Display column vectors in a row for cleaner outputs.
    if (m.cols_ != 1) {
      out << "\n";
    }
  }

  return out;
}

/// Prints a vector of printable things to a stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& data) {
  uint64_t n = data.size();
  for (uint32_t i = 0; i < n; ++i) {
    out << data[i];
    if (i < n - 1) {
      out << ", ";
    }
  }
  return out;
}

/// Generates a vector of the specified dimension full of zeros.
vector<double> Zeros(int count);

#endif //HPSC_MATRIX_H
