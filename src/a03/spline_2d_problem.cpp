
#include <iostream>

#include "gflags/gflags.h"

// TODO(andreib): Stick Eigen stuff in a precompiled header for faster builds!


using namespace std;


int Spline2DExperiment(int argc, char **argv) {
  cout << "Starting 2D spline interpolation experiment." << endl;
  return 0;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return Spline2DExperiment(argc, argv);
}
