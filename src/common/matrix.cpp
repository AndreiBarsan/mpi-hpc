#include "matrix.h"

EMatrix ToEigen(const Matrix<double> &mat) {
  EMatrix res;
  res.resize(mat.rows_, mat.cols_);

  for (uint32_t i = 0; i < mat.rows_; ++i) {
    for (uint32_t j = 0; j < mat.cols_; ++j) {
      res(i, j) = mat(i, j);
    }
  }
  return res;
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
