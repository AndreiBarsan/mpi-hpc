
#ifndef HPSC_DEBOOR_COMMON_H
#define HPSC_DEBOOR_COMMON_H

#include <string>

// TODO(andreib): Move this to another file, e.g., spline_common or whatever.
enum SolverType {
  /// Uses a serial solver built into Eigen.
  kNaiveSparseLU = 0,
  /// Uses a serial DeBoor decomposition.
  kSerialDeBoor,
  /// Uses parallel DeBoor method A.
  kParallelDeBoorA,
  /// Uses parallel DeBoor method B.
  kParallelDeBoorB,
  /// Serial Successive Over-Relaxation (SOR), an iterative method.
  kSerialSOR,
};

SolverType GetSolverType(const std::string &input);

bool IsParallelDeBoor(SolverType solver_type);

#endif //HPSC_DEBOOR_COMMON_H
