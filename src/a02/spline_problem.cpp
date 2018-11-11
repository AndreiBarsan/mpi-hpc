//
// Entry point for solving quadratic spline interpolation.
//
// TODO(andreib): Define from CMake.
#define DEBUG_WITH_EIGEN

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <src/common/utils.h>
#include <thread>
#include <tuple>

#include "gflags/gflags.h"
#include "mpi.h"

#include "matrix.h"
#include "serial_numerical.h"
#include "parallel_numerical.h"

DEFINE_string(out_dir, "../results/spline_output", "The directory where to write experiment results (e.g., for "
                                                   "visualization).");

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

    data.push_back(0);    // This is just for padding.
    data.push_back(4);
    data.push_back(4);
    for (int i = 1; i < n_ + 1; ++i) {
      data.push_back(1);
      data.push_back(6);
      data.push_back(1);
    }
    data.push_back(4);
    data.push_back(4);
    data.push_back(0);    // This is just for padding.

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

SplineProblem BuildCustomProblem(int n) {
  auto function = [](double x) { return 3.0 * sin(x) + sin(3 * x); };
  return SplineProblem("custom", n, function, 0.0, M_PI * 6.0);
}

uint32_t min(uint32_t a, uint32_t b) {
  if (a > b) {
    return b;
  }
  else {
    return a;
  }
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
    Eigen::Matrix<double, Dynamic, 1> x = A_eigen.colPivHouseholderQr().solve(b_eigen);

    // Convert the Eigen matrix to our own type and return.
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
    Matrix<double> b_mat(A.get_n(), 1, b);
    return SolveSerial(A, b_mat);
  }
  else if (method == SolverTypes::kPartitionTwo) {
    Matrix<double> b_mat(A.get_n(), 1, b);
    return SolveParallel(A, b_mat, argc, argv);
  }
  else {
    throw runtime_error("Unsupported solver type.");
  }
}

SplineSolution<double> Solve(const SplineProblem& problem, SolverTypes solver, int argc, char **argv) {
  MPI_SETUP;

  auto A = problem.get_A();
  auto u = problem.get_u();
  cout << "System setup complete." << endl;

  BandMatrix<double> A_cpy(A);
  vector<double> u_cpy = u;
  auto c = SolveSystem(A, u, argc, argv, solver);
  MASTER {
#ifdef DEBUG_WITH_EIGEN
    // Make the master node check the solution using a built-in solver, if it is available.
    auto c_eigen = SolveSystem(A_cpy, u_cpy, argc, argv, SolverTypes::kEigenDense);
    cout << "Eigen solution: " << c_eigen << endl;
    cout << "Our solution:   " << c << endl;
    if (!c.all_close(c_eigen)) {
      throw runtime_error("Sanity check failed! Our solution was different from what Eigen computed.");
    }
#endif
    cout << "Finished computing solution to problem: " << problem.get_full_name() << endl;
  }

  // TODO(andreib): Populate this accordingly after computation complete, including ERROR INFO!
  vector<double> c_vec;
  for(int i =0;i<A.get_n();++i) {
    c_vec.push_back(c(i, 0));
  }
  return SplineSolution<double>(u, c_vec, problem);
}

void Save(const SplineSolution<double> &solution, const string &out_dir) {
  cout << "Will save results to output directory: " << out_dir << endl;
  auto &problem = solution.problem_;
  // These are the points where we plot the interpolated result (and the GT fn).
  auto plot_points = Linspace(problem.a_, problem.b_, 3 * problem.n_ + 1);

  vector<double> gt_y;
  vector<double> interp_y;
  for(double x : plot_points) {
    gt_y.push_back(problem.function_(x));
    interp_y.push_back(solution(x));
  }

  if (! IsDir(out_dir)) {
    if (mkdir(out_dir.c_str(), 0755) == -1) {
      throw runtime_error(Format("Coult not create output dir: %s from working dir %s.", out_dir.c_str(),
                                 GetCWD().c_str()));
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

//  cout << "Interpolation result:" << endl;
//  cout << interp_y << endl;
//  cout << interp_y[interp_y.size() - 2] << " ";
//  cout << interp_y[interp_y.size() - 1] << " ";
//  cout << interp_y[interp_y.size() - 0] << " ";
}

/// A simple test to ensure that we support multiple right-hand sides OK. (And a pentadiagonal matrix.)
void TestMultiRHS() {
  // Bandwidth is 2 so effective bw is 5. Rows have 3, 4, 5, ..., 5, 4, 3 elements.
  BandMatrix<double> A(8, {
    // Solution: pad with zeros in the beginning and end for ease of indexing.
     0, 0, 1, 0, 0,
     0, 0, 2, 0, 0,
     0, 0, 3, 0, 0,
         0, 0, 4, 0, 0,
            0, 0, 5, 0, 0,
               0, 0, 6, 0, 0,
                  0, 0, 7, 0, 0,
                     0, 0, 8, 0, 0
    }, 2);
  Matrix<double> B(8, 3, {
    1, 0, 2,
    2, 0, 2,
    3, 0, 2,
    4, 0, 2,
    5, 0, 2,
    6, 0, 2,
    7, 0, 2,
    8, 0, 2,
  });
  BandMatrix<double> C(5, {
    0, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 0
  });
  cout << A << endl << B << endl << C << endl;

  auto x = SolveSerial(A, B, true);

  Matrix<double> expected_x(8, 3, {
    1, 0, 2,
    1, 0, 1,
    1, 0, 0.666667,
    1, 0, 0.5,
    1, 0, 0.4,
    1, 0, 0.333333,
    1, 0, 0.285714,
    1, 0, 0.25,
  });

  assert (expected_x.all_close(x));
  cout << "Multi-system solver seems to be OK." << endl;
}

int SplineExperiment(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_SETUP;
  vector<int> ns = {30, 62, 126, 254, 510};
//  vector<int> ns = {30}; //, 62, 126, 254, 510};

//  SolverTypes solver = SolverTypes::kPartitionTwo;
  SolverTypes solver = SolverTypes::kCustomSingleThread;
//  MASTER {
//    TestMultiRHS();
//  }

  for (int n : ns) {
    // For both problems, 'Solve' generates the problem matrices and vectors, applies the partitioning to compute the
    // solution, computes maximum errors within each processor's subintervals, and the global errors over all nodes
    // and over 3n+1 points.
    auto problems = {BuildFirstProblem(n), BuildSecondProblem(n), BuildCustomProblem(n)};
    for (const auto &problem : problems) {
      auto solution = Solve(problem, solver, argc, argv);
      MASTER {
        Save(solution, FLAGS_out_dir);
      }
    }
  }

//  system("python ../src/a02/plot_output.py ../results/spline_output/");

  MPI_Finalize();
  return 0;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return SplineExperiment(argc, argv);
}
