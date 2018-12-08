#include "a03/deboor_common.h"

#include <string>

#include "common/utils.h"
#include "deboor_common.h"

SolverType GetSolverType(const std::string &input) {
  if (input == "eigen") {
    return SolverType::kNaiveSparseLU;
  }
  else if (input == "serial-deboor") {
    return SolverType::kSerialDeBoor;
  }
  else if (input == "parallel-deboor-a") {
    return SolverType::kParallelDeBoorA;
  }
  else if (input == "parallel-deboor-b") {
    return SolverType::kParallelDeBoorB;
  }
  else if (input == "sor-natural") {
    return SolverType::kSerialSOR;
  }
  else {
    throw std::runtime_error(Format("Unknown type of solver: []", input.c_str()));
  }
}

bool IsParallelDeBoor(SolverType solver_type) {
  return solver_type == SolverType::kParallelDeBoorA || solver_type == SolverType::kParallelDeBoorB;
}
