
#ifndef HPSC_DEBOOR_COMMON_H
#define HPSC_DEBOOR_COMMON_H

#include <string>

enum SolverType {
  /// Uses a serial solver built into Eigen.
  kNaiveSparseLU = 0,
  /// Uses a serial DeBoor decomposition.
  kSerialDeBoor,
  /// Uses parallel DeBoor method A.
  kParallelDeBoorA,
  /// Uses parallel DeBoor method B.
  kParallelDeBoorB
};

SolverType GetSolverType(const std::string &input);

#endif //HPSC_DEBOOR_COMMON_H
