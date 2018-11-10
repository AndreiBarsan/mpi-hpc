//
// Entry point for solving quadratic spline interpolation.
//

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
#include <tuple>

#include "gflags/gflags.h"

// TODO(andreib): Define from CMake.
#define DEBUG_WITH_EIGEN

#ifdef DEBUG_WITH_EIGEN
#include "Eigen/Eigen"
#endif

#include "mpi.h"

// This is not great practice, but saves me lots of typing.
using namespace std;
using ScalarFunction = function<double(double)>;

enum SolverTypes {
  // A library-provided solver, useful for debugging.
  kEigenDense = 0,
  // The MPI-powered solver implemented for assignment 2.
  kPartitionTwo
};


template<typename T>
class Row {
 public:
  Row(int row_id, const std::vector<T> *data, int bandwidth)
    : row_id_(row_id),
      data_(data),
      n_(data->size() / (bandwidth * 2 + 1) + 1),
      bandwidth_(bandwidth) {}

  T operator[](int col_id) {
    assert(col_id >= 0 && col_id <= n_ - 1);

    if (abs(col_id - row_id_) <= bandwidth_) {
      int off = col_id - row_id_;
      return (*data_)[row_id_ * (bandwidth_ * 2 + 1) + off];
    }
    else {
      return 0;
    }
  }

 private:
  const int bandwidth_;
  const int row_id_;
  const long n_;
  const std::vector<T> *data_;
};


/// Stores data in compact row-major order. Assumes each row has 3 elements except the first and the last.
template<typename T>
class TridiagonalMatrix {

 public:
  TridiagonalMatrix(int n, const std::vector<T> &data) : n_(n), bandwidth_(1), data_(data) {
    assert(data.size() == 3 * n - 2);
  }

  Row<T> operator[](int i) {
    return Row<T>(i, &data_, bandwidth_);
  }

  Row<T> operator[](int i) const {
    return Row<T>(i, &data_, bandwidth_);
  }

  TridiagonalMatrix& operator*(const T& other_scalar) {
    for (T &val : data_) {
      val *= other_scalar;
    }
    return *this;
  }

  int get_n() const {
    return n_;
  }

  /// Returns a dense representation of the data in this matrix.
  std::vector<T> get_dense() const {
    vector<T> result;
    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < n_; ++j) {
        result.push_back(this[i][j]);
      }
    }
    return result;
  }

 private:
  int bandwidth_;
  int n_;
  std::vector<T> data_;


};

/// Useful for printing a matrix as text to a stream. (Includes all zeros.)
template<typename T>
ostream& operator<<(ostream& out, const TridiagonalMatrix<T> &m) {
  int n = m.get_n();

  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      out << m[i][j] << " ";
    }
    out << endl;
  }

  return out;
}

/// Prints a vector of printable things to a stream.
template<typename T>
ostream& operator<<(ostream& out, const vector<T>& data) {
  int n = data.size();
  for (int i = 0; i < n; ++i) {
    out << data[i];
    if (i < n - 1) {
      out << ", ";
    }
  }
  return out;
}


/// Identical functionality to the 'linspace' function from numpy.
vector<double> Linspace(double a, double b, int n) {
  vector<double> res;
  double cur = a;
  // Ensure we reach b exactly without having to do n+1 steps.
  double step_size = (b - a) / (n - 1);
  for (int i = 0; i < n; ++i) {
    res.push_back(cur);
    cur += step_size;
  }
  return res;
}


/// Represents an interpolation problem
class SplineProblem {
 public:
  /// Constructs a quadratic spline interpolation problem.
  /// \param n        The number of equidistant knots. (Will create n+1 knots.)
  /// \param function The ground truth scalar function.
  /// \param a        The start of the interval.
  /// \param b        The end of the interval.
  SplineProblem(int n, ScalarFunction function, double a, double b)
    : n_(n),
      function_(function),
      a_(a),
      b_(b),
      step_size_((b - a) / n)
  { }

  TridiagonalMatrix<double> get_A() const {
    vector<double> data;

    data.push_back(4);
    data.push_back(4);
    for (int i = 1; i < n_ + 1; ++i) {
      data.push_back(1);
      data.push_back(6);
      data.push_back(1);
    }
    data.push_back(4);
    data.push_back(4);

    return TridiagonalMatrix<double>(n_ + 2, data) * (1.0 / 8.0);
  }

  vector<double> get_control_points() const {
    vector<double> knots = Linspace(a_, b_, n_ + 1);

    vector<double> midpoints_and_endpoints;
    midpoints_and_endpoints.reserve(n_ + 2UL);
    midpoints_and_endpoints.push_back(knots[0]);
    for(int i = 1; i < n_ + 1; ++i) {
      midpoints_and_endpoints.push_back((knots[i - 1] + knots[i]) / 2.0);
    }
    midpoints_and_endpoints.push_back(knots[knots.size() - 1]);
    assert(midpoints_and_endpoints.size() == n_ + 2UL);
    return midpoints_and_endpoints;
  }

  vector<double> get_u() const {
    vector<double> u;
    u.reserve(n_ + 2UL);
    for (double &val : get_control_points()) {
      u.push_back(function_(val));
    }
    return u;
  }

  int n_;
  ScalarFunction function_;
  double a_;
  double b_;
  double step_size_;
};

/// Models to resulting solution of a spline problem.
template<typename T>
class SplineSolution {
 public:
  SplineSolution(const vector<T> &coefs, const SplineProblem &problem) : coefs_(coefs), problem_(problem) {}

  T operator()(T x) {
    int i = static_cast<int>(ceil(x / problem_.step_size_));
    cout << "Computed i = " << i << ", n = " << problem_.n_ << endl;
    cout << "x = " << x << ", step size = " << problem_.step_size_ << endl;
    T val = 0;

    if (i > 0) {
      val += coefs_[i - 1] * phi_i(i - 1, x);
    }
    val += coefs_[i] * phi_i(i, x);
    if (i < problem_.n_ + 2) {
      val += coefs_[i + 1] * phi_i(i + 1, x);
    }

    return val;
  }

  const vector<T> coefs_;
  const SplineProblem problem_;
  // TODO include resulting polynomials and error estimates here.
 private:
  T phi_i(int i, T x) {
    assert(i >= 0 && i <= problem_.n_ + 2);
    return phi((x - problem_.a_) / problem_.step_size_ - i + 2);
  }

  T phi(T x) {
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
};

SplineProblem BuildFirstProblem(int n) {
  auto function = [](double x) { return x * x; };
  // TODO(andreib): change back to 0..1
  return SplineProblem(n, function, 0.0, 3.0);
}

SplineProblem BuildSecondProblem(int n) {
  auto function = [](double x) { return x * sin(x); };
  return SplineProblem(n, function, 0.0, M_PI * 12.0);
}

vector<double> SolveSystem(
    const TridiagonalMatrix<double> &A,
    const vector<double> &b,
    SolverTypes method) {

  if (method == SolverTypes::kEigenDense) {
#ifdef DEBUG_WITH_EIGEN
    using namespace Eigen;
    Matrix<double, Dynamic, Dynamic> A_eigen;
    Matrix<double, Dynamic, 1> b_eigen;
    cout << "Solving system using Eigen..." << endl;

    A_eigen.resize(A.get_n(), A.get_n());
    b_eigen.resize(A.get_n(), 1);

    for (int i = 0; i < A.get_n(); ++i) {
      for (int j = 0; j < A.get_n(); ++j) {
        A_eigen(i, j) = A[i][j];
      }
      b_eigen(i) = b[i];
    }

    cout << A_eigen.size() << ", " << A_eigen.rows() << ", " << A_eigen.cols() << endl;
    cout << A_eigen << endl;
    cout << b_eigen << endl;


    cout << "Eigen setup complete." << endl;
    Matrix<double, Dynamic, 1> x = A_eigen.colPivHouseholderQr().solve(b_eigen);
    cout << "Eigen solution computed." << endl;

    // Dirty conversion back to plain old vectors.
    vector<double> res;
    for (int i = 0; i < A.get_n(); ++i) {
      res.push_back(x(i));
    }
    return res;
#else
    throw runtime_error("Requested Eigen solver, but Eigen support is disabled!")
#endif
  }
  else {
    throw runtime_error("Unsupported solver type.");
  }
}

SplineSolution<double> Solve(const SplineProblem& problem) {
  auto A = problem.get_A();
  auto u = problem.get_u();
  cout << "System setup complete." << endl;

  auto c = SolveSystem(A, u, SolverTypes::kEigenDense);
  cout << "Eigen solution: " << c << endl;

  // These are the points where we plot the interpolated result (and the GT fn).
  auto plot_points = Linspace(problem.a_, problem.b_, 3 * problem.n_ + 1);
  cout << "Plot points:" << endl;
  cout << plot_points << endl;

  // TODO(andreib): Populate this accordingly after computation complete.
  SplineSolution<double> solution(c, problem);

  vector<double> gt_y;
  vector<double> interp_y;
  for(double x : plot_points) {
    gt_y.push_back(problem.function_(x));
    interp_y.push_back(solution(x));
  }

  cout << "Finished computing stuff." << endl;

  // TODO make out dir arg so you can use CDF if needed.
  ofstream dump("../results/spline_output/splines-p01.json");
  if (!dump) {
    throw runtime_error("Could not write output.");
  }
  dump << "{" << endl;
  dump << "\t\"control_x\": [" << problem.get_control_points() << "]," << endl;
  dump << "\t\"control_y\": [" << u << "]," << endl;
  dump << "\t\"coefs\":[" << c << "]," << endl,
  dump << "\t\"x\": [" << plot_points << "]," << endl;
  dump << "\t\"gt_y\": [" << gt_y << "]," << endl;
  dump << "\t\"interp_y\": [" << interp_y << "]" << endl;
  dump << "}";

  cout << "Interpolation result:" << endl;
  cout << interp_y << endl;
  cout << interp_y[interp_y.size() - 2] << " ";
  cout << interp_y[interp_y.size() - 1] << " ";
  cout << interp_y[interp_y.size() - 0] << " ";

  // TODO also dump error info!
  return solution;
}

int SplineExperiment(int argc, char **argv) {
  int n = 50;
//  auto problem = BuildFirstProblem(n);
  auto problem = BuildSecondProblem(n);
  auto solution = Solve(problem);

//  system("python ../src/a02/plot_output.py");

  return 0;
}


int main(int argc, char **argv) {
  return SplineExperiment(argc, argv);
}
