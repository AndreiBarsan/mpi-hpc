/**
 *  @file spline_2d_problem.cpp
 *  @brief Entry point and problem definitions for solving 2D quadratic spline interpolation.
 */

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <gflags/gflags.h>
#include <unsupported/Eigen/KroneckerProduct>

#include "common/mpi_helpers.h"
#include "common/mpi_stopwatch.h"
#include "common/serial_numerical.h"
#include "common/utils.h"
#include "a03/deboor_serial.h"
#include "a03/deboor_parallel.h"

// Notes on indexing in the handout:
//  X <---> N points.
//  Y <---> M points.
// N is the first dimension, M is the second dimension.

// Feedback from second assignment's coding part:
//  - not much, but just make sure to account for sparsity in regular operations when computing theoretical
// complexities; one major mistake was you accounted for sparse/triangular matrices when doing e.g.,
// LU/fwd/back-subst, but NOT when doing matrix-matrix and matrix-vector multiplication, which you should have!


DEFINE_string(out_dir, "../results/spline_2d_output", "The directory where to write experiment results (e.g., for "
                                                      "visualization).");
DEFINE_string(method, "eigen", "Name of the method used to solve the spline problem. Allowed values are: eigen (use a "
                               "generic sparse solver built into Eigen), serial-deboor (use a serial DeBoor "
                               "decomposition), parallel-deboor-a (use a parallel double-direction DeBoor "
                               "decomposition), parallel-deboor-b (use a parallel single-direction DeBoor "
                               "decomposition.");
DEFINE_string(problem_sizes, "30, 62, 126, 254, 510", "A comma-separated list of problem sizes to experiment on.");
DEFINE_int32(repeat, 1, "The number of times to repeat each solver run, in order to achieve statistical confidence "
                        "when doing timing estimation.");
DEFINE_bool(dump_result, true, "Whether to dump the solution ouput for visualization.");


// TODO(andreib): Stick Eigen stuff in a precompiled header for faster builds!
// TODO(andreib): Automate LaTeX table generation for an experiment.
// TODO(andreib): Maybe show some of the heatmaps in the assignment report.
// TODO(andreib): Ensure you use -O3 etc. when doing timing.
// TODO(andreib): Check if the (minor) banding artifacts in solution of problem beta are due to a minor offset bug
//                in your solution class.


// TODO(andreib): Group messages together as much as possible.
// TODO(andreib): Experiment with the second function (beta) in problem 2.
// TODO(andreib): Assert errors produced by parallel method are about the same as sequential ones in Q1.
// TODO(andreib): Measure the parallel execution time taken for the solution of the bi-quadratic spline interpolation
// system with the two alternatives. (Do not include the time to calculate the errors.)
// TODO(andreib): When measuring time, measure in chunks so you can see which parts of the method are the most intense.

using namespace std;

/// Represents a 2D scalar-valued function which we use in our 2D interpolation problems.
using Scalar2DFunction = function<double(double, double)>;

/**
 * @brief Returns the control points for the array [a, b], i.e., the mid-points of all n segments plus a and b.
 * @param n The number of intervals to partition [a, b] in.
 * @param a Start of the interval.
 * @param b End of the interval (inclusive).
 * @return An array of (n + 2) knots.
 */
Eigen::ArrayXd GetControlPoints1d(uint32_t n, double a, double b) {
  using namespace Eigen;
  auto knots = ArrayXd::LinSpaced(n + 1, a, b);
  // I am being a little bit paranoid.
  assert(knots[0] == a);
  assert(knots[n] == b);
  ArrayXd control_points(n + 2);
  control_points(0) = knots[0];
  for(uint32_t i = 1; i < n + 1; ++i) {
    control_points(i) = (knots(i - 1) + knots(i)) / 2.0;
  }
  control_points(n + 1) = knots[n];
  return control_points;
}

/// Returns a [sz x sz] coefficient matrix used to build the S and T matrices.
ESMatrix GetCoefMatrix(uint32_t sz) {
  ESMatrix T(sz + 2, sz + 2);
  vector<ET> triplet_list;
  triplet_list.reserve(sz * 3 + 4);
  triplet_list.emplace_back(0, 0, 4);
  triplet_list.emplace_back(0, 1, 4);
  for (uint32_t i = 1; i < sz + 1; ++i) {
    triplet_list.emplace_back(i, i - 1, 1);
    triplet_list.emplace_back(i, i, 6);
    triplet_list.emplace_back(i, i + 1, 1);
  }
  triplet_list.emplace_back(sz + 1, sz, 4);
  triplet_list.emplace_back(sz + 1, sz + 1, 4);

  T.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return T * (1.0 / 8.0);
}

/// Represents a 2D quadratic spline interpolation problem.
class Spline2DProblem {

 public:
  /// Constructs a 2D quadratic spline interpolation problem.
  /// \param name_        An easy to read name for the problem.
  /// \param n_           The number of equidistant knots in the X dimension. (Will create n+1 knots.)
  /// \param m_           The number of equidistant knots in the Y dimension. (Will create m+1 knots.)
  /// \param function_    The ground truth scalar-valued function.
  /// \param a_x_         x-coord of the interval start.
  /// \param a_y_         y-coord of the interval start.
  /// \param b_x_         x-coord of the interval end.
  /// \param b_y_         y-coord of the interval end.
  ///
  /// Note that for consistency with the the assignment handout, the n rows are indexed by x, and the m columns by y.
  /// In the handout, the interval over X has n elements, and the interval over Y has m elements.
  Spline2DProblem(const string &name, uint32_t n, uint32_t m, const Scalar2DFunction &function,
                  double a_x, double a_y, double b_x, double b_y)
      : name_(name), n_(n), m_(m), function_(function), a_x_(a_x), a_y_(a_y), b_x_(b_x),
        b_y_(b_y), step_size_x_((b_x - a_x) / n), step_size_y_((b_y - a_y) / m),
        S(GetCoefMatrix(n)), T(GetCoefMatrix(m)) {}

  /// Returns the coefficient matrix used to solve the spline problem.
  ESMatrix GetA() const {
    return Eigen::kroneckerProduct(S, T);
  }

//  ESMatrix GetS() const {
//    return S;
//  }

//  ESMatrix GetT() const {
//    return GetCoefMatrix(m_);
//  }

  Eigen::MatrixX2d GetControlPoints() const {
    Eigen::ArrayXd x_coord = GetControlPoints1d(n_, a_x_, b_x_);
    Eigen::ArrayXd y_coord = GetControlPoints1d(m_, a_y_, b_y_);
    auto xy_coord = MeshGrid(x_coord, y_coord);
    assert(xy_coord.rows() == ((n_ + 2) * (m_ + 2)));
    return xy_coord;
  }

  /// Returns the right-hand side vector used to solve the spline problem.
  Eigen::VectorXd Getu() const {
    Eigen::MatrixX2d xy_control = GetControlPoints();
    Eigen::VectorXd result(xy_control.rows(), 1);
    for(uint32_t i = 0; i < xy_control.rows(); ++i) {
      result(i) = function_(xy_control(i, 0), xy_control(i, 1));
    }
    assert(xy_control.rows() == result.rows());
    return result;
  }

  string GetFullName() const {
    return Format("problem-%s-%04d", name_.c_str(), n_);
  }

 public:
  const string name_;
  // n_ rows, m_ columns
  const uint32_t n_;
  const uint32_t m_;
  const Scalar2DFunction function_;
  const double a_x_;
  const double a_y_;
  const double b_x_;
  const double b_y_;

  const ESMatrix S;
  const ESMatrix T;

  const double step_size_x_;
  const double step_size_y_;
};

// TODO(andreib): Make timing classes const while still allowing std::accumulate!
/// Contains the timing information collected after the run of a solver.
struct SolverTiming {
  explicit SolverTiming(const Duration &total_duration) : total_duration_(total_duration) {}

  /// Returns a string representation of the timing info.
  virtual std::string ToString() {
    return Format("%.6ld", chrono::microseconds(total_duration_).count());
  }

  Duration total_duration_;
};

/// Contains more detailed timing information produced by the parallel DeBoor solvers.
struct ParallelDeBoorTiming : public SolverTiming {
  ParallelDeBoorTiming(const Duration &init,
                       const Duration &factorization,
                       const Duration &first_stage,
                       const Duration &transpose_stage,
                       const Duration &second_stage)
      : SolverTiming(init + factorization + first_stage + transpose_stage + second_stage),
        init(init),
        factorization(factorization),
        first_stage(first_stage),
        transpose_stage(transpose_stage),
        second_stage(second_stage) {}

  virtual std::string ToString() override {
    return Format("%s,%.6ld,%.6ld,%.6ld,%.6ld,%.6ld",
        SolverTiming::ToString().c_str(),
        chrono::microseconds(init),
        chrono::microseconds(factorization),
        chrono::microseconds(first_stage),
        chrono::microseconds(transpose_stage),
        chrono::microseconds(second_stage)
    );
  }

  Duration init;
  Duration factorization;
  Duration first_stage;
  Duration transpose_stage;
  Duration second_stage;
};

SolverTiming operator+(const SolverTiming &left, const SolverTiming &right) {
  return SolverTiming(left.total_duration_ + right.total_duration_);
}

ParallelDeBoorTiming operator+(const ParallelDeBoorTiming &left, const ParallelDeBoorTiming &right) {
  return ParallelDeBoorTiming(
      left.init + right.init,
      left.factorization + right.factorization,
      left.first_stage + right.first_stage,
      left.transpose_stage + right.transpose_stage,
      left.second_stage + right.second_stage);
}

SolverTiming Aggregate(const std::vector<SolverTiming> &timings) {
  return std::accumulate(timings.cbegin(), timings.cend(), SolverTiming(Duration(0)));
}


/// Contains the errors of a particular solution to a spline problem.
template<typename T>
struct Spline2DErrors {
  Spline2DErrors(double max_over_control_points, double max_over_dense_points)
    : max_over_control_points(max_over_control_points),
      max_over_dense_points(max_over_dense_points) {}

  const double max_over_control_points;
  const double max_over_dense_points;
};

template <typename T>
class Spline2DSolution {
 public:
  Spline2DSolution(const Eigen::MatrixXd &control_vals_,
                   const EMatrix &coefs_,
                   const Spline2DProblem &problem_)
      : control_vals_(control_vals_), coefs_(coefs_), problem_(problem_) {}

  /// Computes the interpolation result at point (x, y).
  T operator()(T x, T y) const {
    auto i = static_cast<int>(ceil(x / problem_.step_size_x_));
    auto j = static_cast<int>(ceil(y / problem_.step_size_y_));
    T val = 0;
    for(int ii = i - 1; ii <= i + 1; ++ii) {
      for(int jj = j - 1; jj <= j + 1; ++jj) {
        double coef = GetCoef(jj, ii);
        if (fabs(coef) > 1e-8) {
          val += coef
              * PhiI(ii, x, problem_.a_x_, problem_.n_, problem_.step_size_x_)
              * PhiI(jj, y, problem_.a_y_, problem_.m_, problem_.step_size_y_);
        }
      }
    }
    return val;
  }

  Spline2DErrors<T> ComputeErrorsAndValidate() const {
    double kMaxControlPointError = 1e-8;
    const Spline2DProblem &problem = this->problem_;
    auto cpoints = problem.GetControlPoints();
    double max_err = GetMaxError(cpoints, problem, *this);
    // Note that these are EXACTLY the points we wish to fit to, so the error should be zero.
    if (max_err > kMaxControlPointError) {
      throw runtime_error(Format("Found unusually large error in a control point. Maximum error over control points "
                                 "was %.10f, larger than the threshold of %.10f.", max_err, kMaxControlPointError));
    }
    auto denser_grid = MeshGrid(
        // Fixed size grid for all n, m, as indicated in the handout.
        GetControlPoints1d(38, problem.a_x_, problem.b_x_),
        GetControlPoints1d(38, problem.a_y_, problem.b_y_)
    );
    double max_err_dense = GetMaxError(denser_grid, problem, *this);
    if (max_err_dense - max_err < -kMaxControlPointError) {
      throw runtime_error(Format("The max error on the dense grid should NOT be smaller than the max error on the "
                                 "control points, but got dense max error %.10f < control point max error %.10f!",
                                 max_err_dense, max_err));
    }
    return Spline2DErrors<T>(max_err, max_err_dense);
  }

  const Eigen::MatrixXd control_vals_;
  const EMatrix coefs_;
  const Spline2DProblem problem_;

 private:
  T GetCoef(int i, int j) const {
    if (i < 0 || j < 0) {
      return 0.0;
    }
    if (i > problem_.n_ + 1 || j > problem_.m_ + 1) {
      return 0.0;
    }
    return coefs_(i, j);
  }
};

Spline2DProblem BuildFirstProblem(uint32_t n, uint32_t m) {
  auto function = [](double x, double y) { return x * x * y * y; };
  return Spline2DProblem("quad", n, m, function, 0.0, 0.0, 1.0, 1.0);
}

Spline2DProblem BuildSecondProblem(uint32_t n, uint32_t m) {
  auto function = [](double x, double y) { return sin(x) * exp(y); };
  return Spline2DProblem("sin-exp", n, m, function, 0.0, 0.0, M_PI * 4, M_PI);
}

/// This is an interesting problem which makes it very obvious visually how a denser grid leads to fewer artifacts.
Spline2DProblem BuildThirdProblem(uint32_t n, uint32_t m) {
  auto function = [](double x, double y) { return sin(x * x) * cos(y * y); };
  return Spline2DProblem("sin(x^2)cos(y^2)", n, m, function, 0.0, 0.0, M_PI * 4, M_PI);
}

Spline2DProblem BuildRosenbrock(uint32_t n, uint32_t m, double a = 1, double b = 100) {
  auto function = [a, b](double x, double y) { return (a - x) * (a - x) + b * (y - x * x) * (y - x * x); };
  return Spline2DProblem("rosenbrock", n, m, function, 0.0, 0.0, 5.0, 5.0);
}

double GetMaxError(const Eigen::MatrixX2d &cpoints, const Spline2DProblem& p, const Spline2DSolution<double>& s) {
  double max_err = -1.0;
  for(int i = 0; i < cpoints.rows(); ++i) {
    double interp_val = s(cpoints(i, 0), cpoints(i, 1));
    double true_val = p.function_(cpoints(i, 0), cpoints(i, 1));

    double err = fabs(interp_val - true_val);
    if (err > max_err) {
      max_err = err;
    }
  }
  return max_err;
}


void Save(const Spline2DSolution<double> &solution, const string &out_dir) {
  cout << "Will save results to output directory: " << out_dir << endl;
  auto &problem = solution.problem_;

  // These are the points where we plot the interpolated result (and the GT fn).
  int count = 3 * problem.n_ + 1;
  if (count < 100) {
    count = 100;
  }

  auto &p = solution.problem_;
  auto denser_grid = MeshGrid(
      GetControlPoints1d(3 * p.n_ + 1, p.a_x_, p.b_x_),
      GetControlPoints1d(3 * p.m_ + 1, p.a_y_, p.b_y_)
  );

  // TODO(andreib): Use Eigen for these.
  vector<double> gt_y;
  vector<double> interp_y;
  for(int i = 0; i < denser_grid.rows(); ++i) {
    gt_y.push_back(problem.function_(denser_grid(i, 0), denser_grid(i, 1)));
    interp_y.push_back(solution(denser_grid(i, 0), denser_grid(i, 1)));
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

  Eigen::IOFormat one_row_fmt(2, 0, ", ", ", ", "", "");
  dump << "{\n"
       << "\t\"m\": " << problem.m_ << ",\n"
       << "\t\"n\": " << problem.n_ << ",\n"
       << "\t\"control_x\": [" << problem.GetControlPoints().format(one_row_fmt) << "],\n"
       << "\t\"control_y\": [" << solution.control_vals_.format(one_row_fmt) << "],\n"
//       << "\t\"coefs\":[" << solution.coefs_.format(one_row_fmt) << "],\n"
//       << "\t\"x\": [" << denser_grid.format(one_row_fmt) << "],\n"
       << "\t\"gt_y\": [" << gt_y << "],\n"
       << "\t\"interp_y\": [" << interp_y << "],\n"
       << "\t\"name\":  \"" << p.GetFullName() << "\"\n"
       << "}"
       << endl;   // Flush the stream.
}

/// Solves the given problem serially using a sparse solver built into Eigen.
/// Useful for checking the correctness of more sophisticated solvers but much slower than DeBoor.
Spline2DSolution<double> SolveNaive(const Spline2DProblem &problem) {
  ESMatrix A = problem.GetA();
  Eigen::VectorXd u = problem.Getu();

  // Note that we ALWAYS compute the solution using the generic Eigen solver so that we can verify our answers.
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  // We'd first need to scale the coef matrix to use this. Needs symmetric positive definite coef matrix.
  // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    throw runtime_error("Could not factorize sparse linear system.");
  }
  Eigen::MatrixXd x = solver.solve(u);
  if (solver.info() != Eigen::Success) {
    throw runtime_error("System factorization OK, but could not solve.");
  }
  EMatrix x_square(x);
  x_square.resize(problem.n_ + 2, problem.m_ + 2);
  return Spline2DSolution<double>(u, x_square, problem);
}

Spline2DSolution<double> SolveSerialDeBoor(const Spline2DProblem &problem, MPIStopwatch &stopwatch) {
  Eigen::MatrixXd deboor_x = DeBoorDecomposition(problem.S, problem.T, problem.Getu(), stopwatch);
  deboor_x.resize(problem.n_ + 2, problem.m_ + 2);
  return Spline2DSolution<double>(problem.Getu(), deboor_x, problem);
}

Spline2DSolution<double> Solve(const Spline2DProblem &problem, SolverType solver_type, MPIStopwatch &stopwatch) {
  switch (solver_type) {
    case kNaiveSparseLU:
      return SolveNaive(problem);
    case kSerialDeBoor:
      return SolveSerialDeBoor(problem, stopwatch);
    case kParallelDeBoorA: {
      Eigen::MatrixXd sol = DeBoorParallelA(problem.S, problem.T, problem.Getu(), stopwatch);
      sol.resize(problem.n_ + 2, problem.m_ + 2);
      return Spline2DSolution<double>(problem.Getu(), sol, problem);
    }
    case kParallelDeBoorB: {
      Eigen::MatrixXd sol = DeBoorParallelB(problem.S, problem.T, problem.Getu(), stopwatch);
      sol.resize(problem.n_ + 2, problem.m_ + 2);
      return Spline2DSolution<double>(problem.Getu(), sol, problem);
    }
    default:
      throw runtime_error(Format("Unknown solver type requested: %d", solver_type));
  }
}

/**
 * @brief Solves the given problem with a reference (but slow) solver and checks the solution match.
 */
void CheckSolution(
    const std::string &solver_name,
    const Spline2DProblem &problem,
    const Spline2DSolution<double> &smart_solution
) {
  Spline2DSolution<double> naive_solution = SolveNaive(problem);
  Eigen::MatrixXd delta = naive_solution.coefs_ - smart_solution.coefs_;
  delta.resize(problem.S.rows(), problem.S.cols());
  double delta_norm = delta.norm();
  const double kDeltaNormEps = 1e-8;
  if (delta_norm > kDeltaNormEps) {
    throw runtime_error(Format("Computed solution coefficients from method [%s] differs from the reference naive "
                               "solution by [%.10lf], more than the threshold epsilon of [%.10lf].",
                               solver_name.c_str(), delta_norm, kDeltaNormEps));
  }
}

void CheckWithSerialDeBoor(
    const std::string &solver_name,
    const Spline2DProblem &problem,
    const Spline2DSolution<double> &parallel_solution
) {
  MPIStopwatch dummy_stopwatch;
  Spline2DSolution<double> serial_solution = SolveSerialDeBoor(problem, dummy_stopwatch);
  Eigen::MatrixXd delta = serial_solution.coefs_ - parallel_solution.coefs_;
  double delta_norm = delta.norm();
  const double kDeltaNormEps = 1e-8;
  if (delta_norm > kDeltaNormEps) {
    throw runtime_error("Delta between serial and parallel DeBoor coefficients too large!");
  }

  auto parallel_err = parallel_solution.ComputeErrorsAndValidate();
  auto serial_err = serial_solution.ComputeErrorsAndValidate();

  const double kErrEps = 1e-8;
  const double parallel_serial_error = fabs(parallel_err.max_over_dense_points - serial_err.max_over_dense_points);
  if (parallel_serial_error > kErrEps) {
    throw runtime_error(Format("Difference between %s solution and serial DeBoor too large (%.10ld)!",
        solver_name, delta));
  }
  else {
    cout << "Parallel max error within " << setprecision(10) << kErrEps << " of serial DeBoor max error!\n";
  }
}

int Spline2DExperiment() {
  MPI_SETUP;
  string solver_name = FLAGS_method;
  int32_t repeat = FLAGS_repeat;
  if (repeat < 1) {
    throw runtime_error("Must set repeat >= 1.");
  }

  ofstream timing_file(Format("%s/timing-%s-%02d-proc-%02d-rep.csv",
      FLAGS_out_dir.c_str(),
      solver_name.c_str(),
      n_procs,
      repeat));
  timing_file << "n,mean_ms,std_ms\n";

  SolverType solver_type = GetSolverType(solver_name);
  vector<int> problem_sizes = ParseCommaSeparatedInts(FLAGS_problem_sizes);
  for (const int& size : problem_sizes) {
//    for (const auto& problem : {BuildFirstProblem(size, size),
//                                BuildSecondProblem(size, size),
//                                BuildThirdProblem(size, size)}) {
    for (const auto& problem : {BuildSecondProblem(size, size)}) {
      MASTER { cout << "Will be solving problem: " << problem.GetFullName() << endl; }

      int q = size / n_procs;
      if (q <= 2 && solver_type == kParallelDeBoorB) {
        MASTER { cout << "Skipping problem size " << size << " for np = " << n_procs << " because it would result in "
                      << "a tridiagonal system too small to solve with parallel partitioning method 2, since we would"
                      << "have q < bandwidth." << endl; };
        continue;
      }

      vector<map<string, Duration>> timings;
      vector<long> full_timings_us;
      // Synchronize everything before we start to benchmark. If we don't do this, our timing will be way off since
      // most nodes will be able to start working right away, EXCEPT the one busy with writing the JSON output of the
      // last problem.
      MPI_Barrier(MPI_COMM_WORLD);
      MPIStopwatch stopwatch; stopwatch.Start();
      auto smart_solution = Solve(problem, solver_type, stopwatch);
      stopwatch.Record("end");
      timings.push_back(stopwatch.GetAllMaxTimesUs());
      full_timings_us.push_back(stopwatch.GetMaxTotalTimeUs().count());

      for(int32_t rep = 1; rep < repeat; ++rep) {
        MPIStopwatch sw; sw.Start();
        auto sol = Solve(problem, solver_type, sw);
        sw.Record("end");
        timings.push_back(sw.GetAllMaxTimesUs());
        full_timings_us.push_back(sw.GetMaxTotalTimeUs().count());

        // TODO(andreib): Average each category.
        MASTER {
          cout << sol.problem_.GetFullName() << ": solved iteration " << rep + 1 << "/" << repeat << ".\n";
//          if (duration_map.size() > 2) {
//            cout << ""
//          }
        }
      }

      MASTER {
        // TODO(andreib): Code for generating LaTeX from this!
        double sum_ms = 0.0;
        double sum_sq_ms = 0.0;
        const int warmup = 1;
        const int cooldown = 1;
        for (int i = warmup; i < full_timings_us.size() - cooldown; ++i) {
          double time_ms = static_cast<double>(full_timings_us[i]) / 1000.0;
          sum_ms += time_ms;
          sum_sq_ms += time_ms * time_ms;
        }
        int count = (full_timings_us.size() - warmup - cooldown);
        double mean = sum_ms / count;
        double std = sqrt(sum_sq_ms / count - mean * mean);
        cout << "Timing: " << mean << "ms, (std = " << std << "ms)" << endl;
        timing_file << size << "," << mean << "," << std << endl;

        if (IsParallelDeBoor(solver_type)) {
          CheckWithSerialDeBoor(solver_name, problem, smart_solution);
        }
        if (FLAGS_dump_result) {
          Save(smart_solution, FLAGS_out_dir);
          cout << "Solution saved as JSON (but not checked yet).\n";
        }
        if (size < 500) {
          cout << "Computing solution using slow method and checking results...\n";
          CheckSolution(solver_name, problem, smart_solution);
          cout << "Solver: " << solver_name << " coefficient check vs. reference solution OK. Checking max error.\n";
        }
        else {
          cout << "Problem too large to check with naive LU solver.\n";
        }
        auto errors = smart_solution.ComputeErrorsAndValidate();
        cout << "Maximum error over control points: " << errors.max_over_control_points << "\n";
        cout << "Maximum error over denser grid: " << errors.max_over_dense_points << "\n";
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  MPI_Init(&argc, &argv);
  int exit_code = Spline2DExperiment();
  MPI_Finalize();
  return exit_code;
}
