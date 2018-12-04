
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

// TODO(andreib): Do we still need this separate enum?
//enum DeBoorMethod {
//  /// This represents "Alternative 1" from the slides.
//      kLinSolveBothDimensions = 0,
//  /// This represents "Alternative 2" from the slides.
//      kLinSolveOneDimension
//};


#endif //HPSC_DEBOOR_COMMON_H
