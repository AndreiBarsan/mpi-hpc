#include "matrix.h"

void ToEigen(const Matrix<double> &mat, EMatrix &out) {
  out.resize(mat.rows_, mat.cols_);
  for (uint32_t i = 0; i < mat.rows_; ++i) {
    for (uint32_t j = 0; j < mat.cols_; ++j) {
      out(i, j) = mat(i, j);
    }
  }
}

EMatrix ToEigen(const Matrix<double> &mat) {
  EMatrix out;
  ToEigen(mat, out);
  return out;
}

Matrix<double> ToMatrix(const EMatrix &eigen) {
  vector<double> data;
  for(int i = 0; i < eigen.rows(); ++i) {
    for(int j = 0; j < eigen.cols(); ++j) {
      data.push_back(eigen(i, j));
    }
  }
  return Matrix<double>(eigen.rows(), eigen.cols(), data);
}

BandMatrix<double> ToTridiagonalMatrix(const ESMatrix &eigen) {
  if (eigen.rows() != eigen.cols()) {
    throw runtime_error("Must pass a square matrix!");
  }
  long n = eigen.rows();
  std::vector<double> data; data.reserve(n * 3);
  data.push_back(0);
  for(long i = 0; i < n; ++i) {
    if (i > 0) {
      data.push_back(eigen.coeff(i, i - 1));
    }
    data.push_back(eigen.coeff(i, i));
    if (i < n - 1) {
      data.push_back(eigen.coeff(i, i + 1));
    }
  }
  data.push_back(0);

  return BandMatrix<double>(n, data);
}

EMatrix ToEigen(const BandMatrix<double> &mat) {
  EMatrix res;
  res.resize(mat.get_n(), mat.get_n());

  for (uint32_t i = 0; i < mat.get_n(); ++i) {
    for (uint32_t j = 0; j < mat.get_n(); ++j) {
      res(i, j) = mat.get(i, j);
    }
  }
  return res;
}

vector<double> Zeros(int count) {
  vector<double> res;
  for(int i = 0; i < count; ++i) {
    res.push_back(0.0);
  }
  return res;
}
