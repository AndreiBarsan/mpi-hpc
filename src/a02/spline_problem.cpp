//
// Entry point for solving 1D quadratic spline interpolation.
//

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <thread>
#include <tuple>

#include "gflags/gflags.h"
#include <Eigen/QR>

#include "common/matrix.h"
#include "common/serial_numerical.h"
#include "common/parallel_numerical.h"
#include "common/utils.h"

DEFINE_string(out_dir, "../results/spline_output", "The directory where to write experiment results (e.g., for "
                                                   "visualization).");

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

/// Represents a quadratic spline interpolation problem.
class SplineProblem {
 public:
  /// Constructs a quadratic spline interpolation problem.
  /// \param name     An easy to read name for the problem.
  /// \param n        The number of equidistant knots. (Will create n+1 knots.)
  /// \param function The ground truth scalar function.
  /// \param a        The start of the interval.
  /// \param b        The end of the interval.
  SplineProblem(const string &name, uint32_t n, const ScalarFunction &function, double a, double b)
    : n_(n),
      function_(function),
      a_(a),
      b_(b),
      step_size_((b - a) / n),
      name_(name)
  { }

  /// Returns the tridiagonal coefficient matrix used in solving the problem.
  BandMatrix<double> GetA() const {
    vector<double> data;

    data.push_back(0);    // This is just for padding.
    data.push_back(4);
    data.push_back(4);
    for (uint32_t i = 1; i < n_ + 1; ++i) {
      data.push_back(1);
      data.push_back(6);
      data.push_back(1);
    }
    data.push_back(4);
    data.push_back(4);
    data.push_back(0);    // This is just for padding.

    return BandMatrix<double>(n_ + 2, data) * (1.0 / 8.0);
  }

  vector<double> GetControlPoints() const {
    vector<double> knots = Linspace(a_, b_, n_ + 1);
    vector<double> midpoints_and_endpoints;
    midpoints_and_endpoints.reserve(n_ + 2UL);
    midpoints_and_endpoints.push_back(knots[0]);
    for(uint32_t i = 1; i < n_ + 1; ++i) {
      midpoints_and_endpoints.push_back((knots[i - 1] + knots[i]) / 2.0);
    }
    midpoints_and_endpoints.push_back(knots[knots.size() - 1]);
    assert(midpoints_and_endpoints.size() == n_ + 2UL);
    return midpoints_and_endpoints;
  }

  /// Returns the right-hand side vector used in solving the problem. (Function value at each control point.)
  vector<double> Getu() const {
    vector<double> u;
    u.reserve(n_ + 2);
    for (double &val : GetControlPoints()) {
      u.push_back(function_(val));
    }
    return u;
  }

  string GetFullName() const {
    return Format("problem-%s-%04d", name_.c_str(), n_);
  }

  uint32_t n_;
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

  /// Computes the interpolation result at point x.
  T operator()(T x) const {
    auto i = static_cast<int>(ceil(x / problem_.step_size_));
    T val = 0;

    if (i > 0) {
      val += coefs_[i - 1] * PhiI(i - 1, x, problem_.a_, problem_.n_, problem_.step_size_);
    }
    val += coefs_[i] * PhiI(i, x, problem_.a_, problem_.n_, problem_.step_size_);

    int ni = problem_.n_ ;
    if (i < ni + 1) {
      val += coefs_[i + 1] * PhiI(i + 1, x, problem_.a_, problem_.n_, problem_.step_size_);
    }
    return val;
  }

  const vector<T> control_y_;
  const vector<T> coefs_;
  const SplineProblem problem_;
  // TODO include resulting polynomials and error estimates here.
};

SplineProblem BuildFirstProblem(uint32_t n) {
  auto function = [](double x) { return x * x; };
  return SplineProblem("quad", n, function, 0.0, 1.0);
}

SplineProblem BuildSecondProblem(uint32_t n) {
  auto function = [](double x) { return sin(x); };
  return SplineProblem("sin", n, function, 0.0, M_PI * 12.0);
}

/// A third custom problem I used for debugging. Not used in the final analysis.
SplineProblem BuildCustomProblem(uint32_t n) {
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
Matrix<double> SolveSystem(BandMatrix<double> &A, vector<double> &b, SolverTypes method) {
  if (method == SolverTypes::kEigenDense) {
#ifdef DEBUG_WITH_EIGEN
    using namespace Eigen;
    Eigen::Matrix<double, Dynamic, Dynamic> A_eigen;
    Eigen::Matrix<double, Dynamic, 1> b_eigen;
    cout << "Solving system using Eigen..." << endl;

    A_eigen.resize(A.get_n(), A.get_n());
    b_eigen.resize(A.get_n(), 1);

    for (uint32_t i = 0; i < A.get_n(); ++i) {
      for (uint32_t j = 0; j < A.get_n(); ++j) {
        A_eigen(i, j) = A.get(i, j);
      }
      b_eigen(i) = b[i];
    }
    Eigen::Matrix<double, Dynamic, 1> x = A_eigen.colPivHouseholderQr().solve(b_eigen);

    // Convert the Eigen matrix to our own type and return.
    vector<double> res;
    for (uint32_t i = 0; i < A.get_n(); ++i) {
      res.push_back(x(i));
    }
    return ::Matrix<double>(A.get_n(), 1, res);
#else
    throw runtime_error("Requested Eigen solver, but Eigen support is disabled!");
#endif
  }
  else if (method == SolverTypes::kCustomSingleThread) {
    Matrix<double> b_mat(A.get_n(), 1, b);
    return SolveSerial(A, b_mat);
  }
  else if (method == SolverTypes::kPartitionTwo) {
    Matrix<double> b_mat(A.get_n(), 1, b);
    return SolveParallel(A, b_mat);
  }
  else {
    throw runtime_error("Unsupported solver type.");
  }
}

SplineSolution<double> Solve(const SplineProblem& problem, SolverTypes solver) {
  MPI_SETUP;
  auto A = problem.GetA();
  auto u = problem.Getu();

  BandMatrix<double> A_cpy(A);
  vector<double> u_cpy = u;
  auto c = SolveSystem(A, u, solver);
  MASTER {
#ifdef DEBUG_WITH_EIGEN
    // Make the master node check the solution using a built-in solver, if it is available.
    auto c_eigen = SolveSystem(A_cpy, u_cpy, SolverTypes::kEigenDense);
    if (!c.all_close(c_eigen)) {
      cerr << "Solution mismatch!" << endl;
      cerr << "Eigen solution: " << c_eigen << endl;
      cerr << "Our solution:   " << c << endl;
      throw runtime_error("Sanity check failed! Our solution was different from what Eigen computed.");
    }
    else {
      cout << "[OK] I computed the solution again in a naive way with Eigen, and the result matched using an epsilon "
              "of " << 1e-6 << "." << endl;
    }
#else
    cout << "Eigen unavailable, so NOT checking solution correctness. You are on your own!" << endl;
#endif
    cout << "Finished computing solution to problem: " << problem.GetFullName() << endl;
  }

  vector<double> c_vec;
  for(uint32_t i = 0; i < A.get_n(); ++i) {
    c_vec.push_back(c(i, 0));
  }
  return SplineSolution<double>(u, c_vec, problem);
}

void Save(const SplineSolution<double> &solution, const string &out_dir) {
  cout << "Will save results to output directory: " << out_dir << endl;
  auto &problem = solution.problem_;
  // These are the points where we plot the interpolated result (and the GT fn).
  int count = 3 * problem.n_ + 1;
  if (count < 100) {
    count = 100;
  }
  auto plot_points = Linspace(problem.a_, problem.b_, count);

  vector<double> gt_y;
  vector<double> interp_y;
  for(double x : plot_points) {
    gt_y.push_back(problem.function_(x));
    interp_y.push_back(solution(x));
  }

  if (! IsDir(out_dir)) {
    if (mkdir(out_dir.c_str(), 0755) == -1) {
      throw runtime_error(Format("Could not create output dir: %s from working dir %s.", out_dir.c_str(),
                                 GetCWD().c_str()));
    }
    else {
      cout << "Created output directory: " << out_dir << endl;
    }
  }

  // Poor man's JSON dumping. Makes it super easy to load the results in Python and plot.
  ofstream dump(Format("%s/output-%s.json", out_dir.c_str(), problem.GetFullName().c_str()));
  if (!dump) {
    throw runtime_error("Could not write output.");
  }
  dump << "{" << endl;
  dump << "\t\"control_x\": [" << problem.GetControlPoints() << "]," << endl;
  dump << "\t\"control_y\": [" << solution.control_y_ << "]," << endl;
  dump << "\t\"coefs\":[" << solution.coefs_ << "]," << endl,
  dump << "\t\"x\": [" << plot_points << "]," << endl;
  dump << "\t\"gt_y\": [" << gt_y << "]," << endl;
  dump << "\t\"interp_y\": [" << interp_y << "]" << endl;
  dump << "}";
}

/// A simple test to ensure that we support multiple right-hand sides OK. (And a pentadiagonal matrix.)
int TestMultiRHS() {
  MPI_SETUP;
  BandMatrix<double> A(8, {
    // Solution: pad with zeros in the beginning and end for ease of indexing.
      0, 1, 0,
        0, 2, 0,
          0, 3, 0,
             0, 4, 0,
               0, 5, 0,
                   0, 6, 0,
                      0, 7, 0,
                         0, 8, 0
    }, 1);
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
  Matrix<double> x_para = SolveParallel(A, B);

  cout << "Return from SolveParallel breh!" << endl;

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

  // If we pad the system naively, do we get the same solution?
  BandMatrix<double> A_pad(10, {
      0, 0, 1, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 2, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 4, 0, 0,
      0, 0, 5, 0, 0,
      0, 0, 6, 0, 0,
      0, 0, 7, 0, 0,
      0, 0, 8, 0, 0,
      0, 0, 1, 0, 0,
  }, 2);
  Matrix<double> B_pad(10, 3, {
      0, 0, 0,
      1, 0, 2,
      2, 0, 2,
      3, 0, 2,
      4, 0, 2,
      5, 0, 2,
      6, 0, 2,
      7, 0, 2,
      8, 0, 2,
      0, 0, 0,
  });

//  assert (expected_x.all_close(x));
  cout << endl;
  MPI_Barrier(MPI_COMM_WORLD);
  cout << endl;

  MASTER {
    double delta = fabs((expected_x - x_para).norm());
    if (delta > 1e-5) {
      cerr << "Expected:" << endl;
      stringstream ss_e;
      ss_e << expected_x;
      cerr << ss_e.str() << endl;
      cerr << "Actual:" << endl;
      stringstream ss_a;
      ss_a << x_para;
      cerr << ss_a.str() << endl;

      throw runtime_error(Format("Incorrect solution. Delta was: %.8f.", delta));
    }
  }
//  assert(x.all_close(x_para));
  cout << "Multi-system solvers seem to be OK." << endl;

//  auto x_pad = SolveSerial(A_pad, B_pad, true);
//  cout << x << endl;
//  cout << x_pad << endl;
  return 0;
}

/// Computes the max abs error of the solution on the specified points of the problem.
double MaxAbsError(const vector<double> &points, const SplineProblem &problem, const SplineSolution<double> &solution) {
  double max_error = -1.0;
  for (double x_i : points) {
    double q_i = solution(x_i);
    double u_i = problem.function_(x_i);
    double abs_error = fabs(q_i - u_i);
    if (abs_error > max_error) {
      max_error = abs_error;
    }
  }
  return max_error;
}

double MaxAbsErrorParallel(
    const vector<double> &points,
    const SplineProblem &problem,
    const SplineSolution<double> &solution
) {
  MPI_SETUP;
  int chunk_size = (points.size() / n_procs) + 1;
  int my_start = local_id * chunk_size;
  int my_count = chunk_size;
  cout << "MAEP: " << local_id << "; start = " << my_start << ", end = " << my_start + my_count << ", size = "
       << points.size() << endl;

  double max_error = -1.0;
  for (int i = my_start; i < my_start + my_count; ++i) {
    if (i < points.size()) {
      double x_i = points[i];
      double q_i = solution(x_i);
      double u_i = problem.function_(x_i);
      double abs_error = fabs(q_i - u_i);
      if (abs_error > max_error) {
        max_error = abs_error;
      }
    }
  }

  double sendbuf = max_error;
  double recvbuf = 0.0;
  MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return recvbuf;
}

int SplineExperiment() {
  MPI_SETUP;
  vector<uint32_t> ns = {14, 30, 62, 126, 254, 510};
//  vector<uint32_t> ns = {62, 126, 254, 510};

  SolverTypes solver = SolverTypes::kPartitionTwo;
//  SolverTypes solver = SolverTypes::kEigenDense;

  for (uint32_t n : ns) {
    // For both problems, 'Solve' generates the problem matrices and vectors, applies the partitioning to compute the
    // solution, computes maximum errors within each processor's subintervals, and the global errors over all nodes
    // and over 3n+1 points.
    auto problems = {BuildFirstProblem(n), BuildSecondProblem(n)};
    for (const auto &problem : problems) {
      MASTER {
        cout << endl << endl << "Solving " << problem.GetFullName() << "..." << endl;
      }
      auto solution = Solve(problem, solver);

      MPI_Barrier(MPI_COMM_WORLD);
      MASTER {
        cout << "Computing error statistics." << endl;
      }

      auto knots = Linspace(problem.a_, problem.b_, problem.n_ + 1);
      auto denser_pts = Linspace(problem.a_, problem.b_, 3 * problem.n_ + 1);
      // Initial code I used to compute erorrs sequentially.
//      double max_knot_error = -1.0;
//      double max_dense_error = -1.0;
//      MASTER {
//        max_knot_error = MaxAbsError(knots, problem, solution);
//        cout << "problem: " << problem.GetFullName() << " (n = " << n << "), max error on the (n + 1) knots = "
//             << max_knot_error << endl;
//
//        max_dense_error = MaxAbsError(denser_pts, problem, solution);
//        cout << "problem: " << problem.GetFullName() << " (n = " << n << "), max error on the (3n + 1) pts = "
//             << max_dense_error << endl;
//      }

      double mke_p = MaxAbsErrorParallel(knots, problem, solution);
      double mde_p = MaxAbsErrorParallel(denser_pts, problem, solution);
      MASTER {
        cout << "problem: " << problem.GetFullName() << " (n = " << n << "), max error on the (n + 1) knots = "
             << mke_p << endl;
        cout << "problem: " << problem.GetFullName() << " (n = " << n << "), max error on the (3n + 1) pts = "
             << mde_p << endl;

        // Some sanity checks.
        assert(mde_p - mke_p > -1e-8);
        assert(n < 75 || mke_p < 1e-4);
      };

//      MASTER {
//        double knot_err_err = fabs(mke_p - max_knot_error);
//        double dense_err_err = fabs(mde_p - max_dense_error);
//        if (knot_err_err > 1e-8) {
//          cout << "Error estimating knot error: " << knot_err_err << endl;
//          cout << mke_p << " vs. " << max_knot_error << endl;
//          throw runtime_error("");
//        }
//        cout << "Parallel knot errors computed OK." << endl;
//
//        if (dense_err_err > 1e-8) {
//          cout << "Error estimating dense error: " << dense_err_err << endl;
//          cout << mde_p << " vs. " << max_dense_error << endl;
//          throw runtime_error("");
//        }
//        cout << "Parallel dense errors computed OK!" << endl;
//      };

      MASTER {
        Save(solution, FLAGS_out_dir);
      }
    }
  }
  return 0;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  MPI_Init(&argc, &argv);
  int exit_code = SplineExperiment();
//  int exit_code = TestMultiRHS();
  MPI_Finalize();
  return exit_code;
}
