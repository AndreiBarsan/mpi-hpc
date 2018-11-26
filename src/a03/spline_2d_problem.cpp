
#include <iostream>
#include <functional>

#include "gflags/gflags.h"

// TODO(andreib): Stick Eigen stuff in a precompiled header for faster builds!


using namespace std;


using Scalar2DFunction = function<double(double, double)>;


class Spline2DProblem {

    // n_ rows, m_ columns
    const uint32_t n_;
    const uint32_t m_;
    const double a_x_;
    const double a_y_;
    const double b_x_;
    const double b_y_;
    const Scalar2DFunction function_;

    const double step_size_x_;
    const double step_size_y_;
};


Spline2DProblem BuildFirstProblem(uint32_t n, uint32_t m) {
  auto function = [](double x, double y) { return x * x * y * y; };
  return Spline2DProblem("quad", n, function, 0.0, 0.0, 1.0, 1.0);
}

Spline2DProblem BuildSecondProblem(uint32_t n, uint32_t m) {
  auto function = [](double x) { return sin(x); };
  return SplineProblem("sin", n, function, 0.0, M_PI * 12.0);
}


int Spline2DExperiment(int argc, char **argv) {
  cout << "Starting 2D spline interpolation experiment." << endl;
  return 0;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return Spline2DExperiment(argc, argv);
}
