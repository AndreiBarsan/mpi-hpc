//
// Very ugly but reasonably efficient matrix classes.
//

#ifndef HPSC_MATRIX_H
#define HPSC_MATRIX_H

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "Eigen/Eigen"
using EMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using namespace std;

/// A quick and dirty row-major dense matrix class.
template<typename T>
class Matrix {
 public:
    Matrix(uint32_t long n, uint32_t m, const std::vector<T> &data)
        : rows_(n), cols_(m), data_(data)
    {
      assert (n * m == data.size());
    }

    T& operator()(int n, int m) {
      assert (n >= 0 && n < rows_);
      assert (m >= 0 && m < cols_);
      return data_[n * cols_ + m];
    }

    T operator()(int n, int m) const {
      assert (n >= 0 && n < rows_);
      assert (m >= 0 && m < cols_);
      return data_[n * cols_ + m];
    }

    bool all_close(const Matrix<T> &other) const {
      assert (rows_ == other.rows_ && cols_ == other.cols_);
      T epsilon = 1e-4;
      for(int i = 0; i < rows_; ++i) {
        for(int j = 0; j < cols_; ++j) {
          if (std::fabs((*this)(i, j) - other(i, j) > epsilon)) {
            return false;
          }
        }
      }
      return true;
    }

    int write_raw_rows(int row_start, int row_end, T *out, int offset) const {
      int idx = offset;
      for(int i = row_start; i < row_end; ++i) {
        for(int j = 0; j < cols_; ++j) {
          out[idx++] = data_[i * cols_ + j];
        }
      }
      return idx;
    }

    int write_raw(T *out, int offset) const {
      return write_raw_rows(0, rows_, out, offset);
    }

    int set_from(T *raw, int offset = 0) {
      int cur_offset = offset;
      for(int i = 0; i < rows_; ++i) {
        for(int j = 0; j < cols_; ++j) {
          int idx = i * cols_ + j;
          data_[idx] = raw[cur_offset++];
        }
      }
      return cur_offset;
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
    assert(row_id >= 0 && row_id < n_);
    assert(col_id >= 0 && col_id < n_);

    if (abs(col_id - row_id) <= bandwidth_) {
      return data_.at(index_(row_id, col_id));
    }
    else {
      throw std::out_of_range("Cannot access off-banded-diagonal element in non-const way.");
    }
  }

  // TODO(andreib): Make this more consistent.
  T get(int row_id, int col_id) const {
    assert(row_id >= 0 && row_id < n_);
    assert(col_id >= 0 && col_id < n_);

    if (abs(col_id - row_id) <= bandwidth_) {
      int off = col_id - row_id;
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
    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < n_; ++j) {
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
  int write_raw_rows(int row_start, int row_end, T *out, int offset) const {
    int idx = offset;
    int effective = bandwidth_ * 2 + 1;
    for(int i = row_start; i < row_end; ++i) {
      for (int raw_col = 0; raw_col < effective; ++raw_col) {
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

bool all_close(const std::vector<double> &left, const std::vector<double> &right) {
  double epsilon = 1e-6;
  for(uint32_t i = 0; i < left.size(); ++i) {
    if (fabs(left[i] - right[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

EMatrix ToEigen(const Matrix<double> &mat) {
  EMatrix res;
  res.resize(mat.rows_, mat.cols_);

  for (uint32_t i = 0; i < mat.rows_; ++i) {
    for (uint32_t j = 0; j < mat.cols_; ++j) {
      res(i, j) = mat(i, j);
    }
  }
  return res;
};

Matrix<double> ToMatrix(const EMatrix &eigen) {
  vector<double> data;
  for(int i = 0; i < eigen.rows(); ++i) {
    for(int j = 0; j < eigen.cols(); ++j) {
      data.push_back(eigen(i, j));
    }
  }
  return Matrix<double>(eigen.rows(), eigen.cols(), data);
}

EMatrix ToEigen(const BandMatrix<double> &mat) {
  EMatrix res;
  res.resize(mat.get_n(), mat.get_n());

  for (int i = 0; i < mat.get_n(); ++i) {
    for (int j = 0; j < mat.get_n(); ++j) {
      res(i, j) = mat.get(i, j);
    }
  }
  return res;
};



template<typename T>
Matrix<T> operator*(const Matrix<T> &left, const Matrix<T> &right) {
  assert(left.cols_ == right.rows_);
  std::vector<T> res_data;
  for(int i = 0; i < left.rows_ * right.cols_; ++i) {
    res_data.push_back(0.0);
  }

  Matrix<T> result(left.rows_, right.cols_, res_data);
  for(int i = 0; i < left.rows_; ++i) {
    for(int j = 0; j < right.cols_; ++j) {
      for(int k = 0; k < left.cols_; ++k) {
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
  for(int i = 0; i < left.rows_; ++i) {
    for(int j = 0; j < left.cols_; ++j) {
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
  for(int i = 0; i < left.rows_; ++i) {
    for(int j = 0; j < left.cols_; ++j) {
      result(i, j) += right(i, j);
    }
  }
  return result;
}

/// Useful for printing a matrix as text to a stream. (Includes all zeros.)
template<typename T>
std::ostream& operator<<(std::ostream& out, const BandMatrix<T> &m) {
  int n = m.get_n();

  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      out << m.get(i, j) << " ";
    }
    out << std::endl;
  }

  return out;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T> &m) {
  for(int i = 0; i < m.rows_; ++i) {
    for(int j = 0; j < m.cols_; ++j) {
      out << m(i, j) << ", ";
    }
    // Display column vectors in a row for cleaner outputs.
    if (m.cols_ != 1) {
      out << std::endl;
    }
  }

  return out;
}

/// Prints a vector of printable things to a stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& data) {
  int n = data.size();
  for (int i = 0; i < n; ++i) {
    out << data[i];
    if (i < n - 1) {
      out << ", ";
    }
  }
  return out;
}

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <fstream>
#include <functional>
#include <src/common/utils.h>
#include <thread>
#include "Eigen/Eigen"

#endif //HPSC_MATRIX_H
