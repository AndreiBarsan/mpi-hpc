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
  // A custom implemented single-threaded LU solver. Building block for the MPI partitioned method.
  kCustomSingleThread,
  // The MPI-powered solver implemented for assignment 2.
  kPartitionTwo
};


bool all_close(const vector<double> &left, const vector<double> &right) {
  double epsilon = 1e-6;
  for(int i = 0; i < left.size(); ++i) {
    if (fabs(left[i] - right[i]) > epsilon) {
      return false;
    }
  }
  return true;
}


/// A quick and dirty row-major dense matrix class.
template<typename T>
class Matrix {
 public:
    Matrix(uint32_t long n, uint32_t m, const vector<T> &data)
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
      T epsilon = 1e-6;   // LoL hack
      for(int i = 0; i < rows_; ++i) {
        for(int j = 0; j < cols_; ++j) {
          if (fabs((*this)(i, j) - other(i, j) > epsilon)) {
            return false;
          }
        }
      }
      return true;
    }

 public:
  const uint32_t rows_, cols_;

 private:
  vector<T> data_;
};


/// Implements a square banded matrix. Stores data in compact row-major order. Assumes each row has 2 * band + 1
/// elements except the first and the last 'band'.
///
/// While the original problem for quadratic spline interpolation is just tridiagonal, the final reduced system is
/// banded with a wider band, so we do need to implement our matrices to support this.
template<typename T>
class BandMatrix {

 public:
  BandMatrix(uint32_t n, const std::vector<T> &data) : n_(n), bandwidth_(1UL), data_(data) {
    // Check that we got the right number of elements.
    uint32_t effective_bw = 2 * bandwidth_ + 1;
    uint32_t missing_at_edge = (effective_bw - 2) * (effective_bw - 1) / 2;
    assert(data.size() == (bandwidth_ * 2 + 1) * n - 2 * missing_at_edge);
  }

//  BandRow<T> operator[](int i) {
//    return BandRow<T>(i, &data_, n_, bandwidth_);
//  }
//
//  BandRow<T> operator[](int i) const {
//    return BandRow<T>(i, &data_, n_, bandwidth_);
//  }

  T& operator()(int row_id, int col_id) {
    assert(row_id >= 0 && row_id < n_);
    assert(col_id >= 0 && col_id < n_);

    if (abs(col_id - row_id) <= bandwidth_) {
      int off = col_id - row_id;
      return data_.at(row_id * (bandwidth_ * 2 + 1) + off);
    }
    else {
      throw runtime_error("Cannot access off-banded-diagonal element in non-const way.");
    }
  }

  // TODO(andreib): Make this more consistent.
  T get(int row_id, int col_id) const {
    assert(row_id >= 0 && row_id < n_);
    assert(col_id >= 0 && col_id < n_);

    if (abs(col_id - row_id) <= bandwidth_) {
      int off = col_id - row_id;
      return data_[row_id * (bandwidth_ * 2 + 1) + off];
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
    vector<T> result_data;
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

 private:
  uint32_t bandwidth_;
  uint32_t n_;
  std::vector<T> data_;
};

template<typename T>
Matrix<T> operator*(const Matrix<T> &left, const Matrix<T> &right) {
  assert(left.cols_ == right.rows_);
  vector<T> res_data;
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

/// Useful for printing a matrix as text to a stream. (Includes all zeros.)
template<typename T>
ostream& operator<<(ostream& out, const BandMatrix<T> &m) {
  int n = m.get_n();

  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      out << m.get(i, j) << " ";
    }
    out << endl;
  }

  return out;
}

template<typename T>
ostream& operator<<(ostream& out, const Matrix<T> &m) {
  for(int i = 0; i < m.rows_; ++i) {
    for(int j = 0; j < m.cols_; ++j) {
      out << m(i, j) << " ";
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
  /// \param name     An easy to read name for the problem.
  /// \param n        The number of equidistant knots. (Will create n+1 knots.)
  /// \param function The ground truth scalar function.
  /// \param a        The start of the interval.
  /// \param b        The end of the interval.
  SplineProblem(const string &name, int n, ScalarFunction function, double a, double b)
    : n_(n),
      function_(function),
      a_(a),
      b_(b),
      step_size_((b - a) / n),
      name_(name)
  { }

  BandMatrix<double> get_A() const {
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

    return BandMatrix<double>(n_ + 2, data) * (1.0 / 8.0);
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

  string get_full_name() const {
    return Format("problem-%s-%04d", name_.c_str(), n_);
  }

  int n_;
  ScalarFunction function_;
  double a_;
  double b_;
  double step_size_;
  string name_;
};

/// Models to resulting solution of a spline problem.
template<typename T>
class SplineSolution {
 public:
  SplineSolution(const vector<T> &control_y, const vector<T> &coefs, const SplineProblem &problem)
      : control_y_(control_y),
        coefs_(coefs),
        problem_(problem) {}

  T operator()(T x) const {
    /// Computes the interpolation result at point x.
    auto i = static_cast<int>(ceil(x / problem_.step_size_));
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

  const vector<T> control_y_;
  const vector<T> coefs_;
  const SplineProblem problem_;
  // TODO include resulting polynomials and error estimates here.
 private:
  T phi_i(int i, T x) const {
    assert(i >= 0 && i <= problem_.n_ + 2);
    return phi((x - problem_.a_) / problem_.step_size_ - i + 2);
  }

  T phi(T x) const {
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
  return SplineProblem("quad", n, function, 0.0, 1.0);
}

SplineProblem BuildSecondProblem(int n) {
  auto function = [](double x) { return sin(x); };
  return SplineProblem("sin", n, function, 0.0, M_PI * 12.0);
}

uint32_t min(uint32_t a, uint32_t b) {
  if (a > b) {
    return b;
  }
  else {
    return a;
  }
}

/// Performs LU factorization of the banded matrix A, in-place.
void BandedLUFactorization(BandMatrix<double> &A, bool check_lu) {
  // For debugging
  Matrix<double> A_orig = A.get_dense();
  uint32_t n = A.get_n();

  uint32_t l = A.get_bandwidth();
  uint32_t u = A.get_bandwidth();
  int bw = l + u + 1;

  for (uint32_t k = 0; k < n - 1; ++k) {
    for (uint32_t i = k + 1; i <= min(k + l, n - 1); ++i) {
      A(i, k) = A(i, k) / A(k, k);

      for (uint32_t j = k + 1; j <= min(k + u, n - 1); ++j) {
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
    for (int i = 0; i < n; ++i) {
      lower(i, i) = 1.0;
    }
    Matrix<double> upper(n, n, upper_data);
    bool all_close = A_orig.all_close(lower * upper);
    if (!all_close) {
      throw runtime_error("LU-decomposition is incorrect! Get a refund! ;)");
    }
  }
}

/*
 * Ax = b
 * LU-decompose A, s.t. A = LU.
 * LUx = b
 * Solve with forward substitution:
 * Lz = b
 *
 * Then, using the z, perform the second (back)substitution:
 * Ux = z
 */
/// Solves the given banded linear system in-place using a LU factorization.
/// Note: Destroys A and b by performing the factorization and substitutions in-place.
Matrix<double> SolveCustom(BandMatrix<double> &A, Matrix<double> &b, bool check_lu = false) {
  // TODO(andreib): Update methods to support solving MULTIPLE systems!
  uint32_t n = A.get_n();
  assert(b.rows_ == n);

  cout << "Will solve " << b.cols_ << " linear systems." << endl;
  BandedLUFactorization(A, check_lu);

  // Perform forward substitution to find intermediate result z.
  Matrix<double> z(b);
  int bw = A.get_bandwidth();
  for (int i = 0; i < n; ++i) {
    for(int j = max(0, i - bw); j <= i - 1; ++j) {
      b(i, 0) = b(i, 0) - A(i, j) * z(j, 0);
    }
    z(i, 0) = b(i, 0); // / A(i, i);  // No divide because lower always has a 1 on the diagonal!
  }

  // Perform backsubstitution
  // We store our output here and the rhs is z.
  Matrix<double> x(b);

  for (int j = n - 1; j >= 0; --j) {
    x(j, 0) = z(j, 0) / A(j, j);        // the upper matrix has non-ones on diag, so we DO need to divide!
    for (int i = max(0, j - bw); i <= max(0, j - 1); i++) {
      z(i, 0) = z(i, 0) - A(i, j) * x(j, 0);
    }
  }

  return x;
}


/// Solves a linear banded system using the specified method.
/// We need the (ugly) argc and argv args for the MPI case.
Matrix<double> SolveSystem(
    BandMatrix<double> &A,
    vector<double> &b,
    int argc,
    char **argv,
    SolverTypes method) {

  if (method == SolverTypes::kEigenDense) {
#ifdef DEBUG_WITH_EIGEN
    using namespace Eigen;
    Eigen::Matrix<double, Dynamic, Dynamic> A_eigen;
    Eigen::Matrix<double, Dynamic, 1> b_eigen;
    cout << "Solving system using Eigen..." << endl;

    A_eigen.resize(A.get_n(), A.get_n());
    b_eigen.resize(A.get_n(), 1);

    for (int i = 0; i < A.get_n(); ++i) {
      for (int j = 0; j < A.get_n(); ++j) {
        A_eigen(i, j) = A.get(i, j);
      }
      b_eigen(i) = b[i];
    }

    cout << "Eigen setup complete." << endl;
    Eigen::Matrix<double, Dynamic, 1> x = A_eigen.colPivHouseholderQr().solve(b_eigen);
    cout << "Eigen solution computed." << endl;

    // Dirty conversion back to plain old vectors.
    vector<double> res;
    for (int i = 0; i < A.get_n(); ++i) {
      res.push_back(x(i));
    }
    return ::Matrix<double>(A.get_n(), 1, res);
#else
    throw runtime_error("Requested Eigen solver, but Eigen support is disabled!")
#endif
  }
  else if (method == SolverTypes::kCustomSingleThread) {
    ::Matrix<double> b_mat(A.get_n(), 1, b);
    return SolveCustom(A, b_mat);
  }
  else {
    throw runtime_error("Unsupported solver type.");
  }
}

SplineSolution<double> Solve(const SplineProblem& problem, int argc, char **argv) {
  auto A = problem.get_A();
  auto u = problem.get_u();
  cout << "System setup complete." << endl;

  BandMatrix<double> A_cpy(A);
  vector<double> u_cpy = u;
  auto c = SolveSystem(A, u, argc, argv, SolverTypes::kCustomSingleThread);
#ifdef DEBUG_WITH_EIGEN
  auto c_eigen = SolveSystem(A_cpy, u_cpy, argc, argv, SolverTypes::kEigenDense);
  cout << "Eigen solution: " << c_eigen << endl;
  cout << "Our solution:   " << c << endl;
  if (! c.all_close(c_eigen)) {
    throw runtime_error("Sanity check failed! Our solution was different from what Eigen computed.");
  }
#endif

  cout << "Finished computing solution to problem: " << problem.get_full_name() << endl;

  // TODO(andreib): Populate this accordingly after computation complete, including ERROR INFO!
  vector<double> c_vec;
  for(int i =0;i<A.get_n();++i) {
    c_vec.push_back(c(i, 0));
  }
  return SplineSolution<double>(u, c_vec, problem);
}

void Save(const SplineSolution<double> &solution) {
  // These are the points where we plot the interpolated result (and the GT fn).
  auto &problem = solution.problem_;
  auto plot_points = Linspace(problem.a_, problem.b_, 3 * problem.n_ + 1);

  vector<double> gt_y;
  vector<double> interp_y;
  for(double x : plot_points) {
    gt_y.push_back(problem.function_(x));
    interp_y.push_back(solution(x));
  }

  // TODO make out dir arg so you can use CDF if needed.
  string out_dir = "../results/spline_output";
  if (! IsDir(out_dir)) {
    if (mkdir(out_dir.c_str(), 0755) == -1) {
      throw runtime_error("Coult not create output dir.");
    }
    else {
      cout << "Created output directory: " << out_dir << endl;
    }
  }

  // Poor man's JSON dumping. Makes it super easy to load the results in Python and plot.
  ofstream dump(Format("%s/output-%s.json", out_dir.c_str(), problem.get_full_name().c_str()));
  if (!dump) {
    throw runtime_error("Could not write output.");
  }
  dump << "{" << endl;
  dump << "\t\"control_x\": [" << problem.get_control_points() << "]," << endl;
  dump << "\t\"control_y\": [" << solution.control_y_ << "]," << endl;
  dump << "\t\"coefs\":[" << solution.coefs_ << "]," << endl,
  dump << "\t\"x\": [" << plot_points << "]," << endl;
  dump << "\t\"gt_y\": [" << gt_y << "]," << endl;
  dump << "\t\"interp_y\": [" << interp_y << "]" << endl;
  dump << "}";

  cout << "Interpolation result:" << endl;
  cout << interp_y << endl;
  cout << interp_y[interp_y.size() - 2] << " ";
  cout << interp_y[interp_y.size() - 1] << " ";
  cout << interp_y[interp_y.size() - 0] << " ";
}

int SplineExperiment(int argc, char **argv) {
  vector<int> ns = {30, 62, 126, 254, 510};

  int processor_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
//  vector<int> ns = {30};

  // TODO(andreib): If time, write proper test for this.
//  vector<double> a_d = {1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0};
//  vector<double> b_d = Linspace(1, 9, 9);
//  Matrix<double> a(3, 3, a_d);
//  Matrix<double> b(3, 3, b_d);
//  cout << a << endl << b << endl;
//  cout << a * b << endl;

  for (int n : ns) {
    // For both problems, 'Solve' generates the problem matrices and vectors, applies the partitioning to compute the
    // solution, computes maximum errors within each processor's subintervals, and the global errors over all nodes
    // and over 3n+1 points.
    auto solution_a = Solve(BuildFirstProblem(n), argc, argv);
    if (0 == processor_rank) {
      Save(solution_a);
    }
    auto solution_b = Solve(BuildSecondProblem(n), argc, argv);
    if (0 == processor_rank) {
      Save(solution_b);
    }
  }

//  system("python ../src/a02/plot_output.py ../results/spline_output/");

  return 0;
}


int main(int argc, char **argv) {
  return SplineExperiment(argc, argv);
}
