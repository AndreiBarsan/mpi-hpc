#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>

#include "mpi.h"

/// Splits a string using the given delimiter.
/// Source: https://stackoverflow.com/a/236803/1055295
template<typename Out>
void Split(const std::string &s, char delim, Out result);

std::vector<std::string> Split(const std::string &s, char delim);

/// Checks whether the path exists.
bool PathExists(const std::string &path);

/// Checks whether the path exists and is a directory.
bool IsDir(const std::string &path);

#define MPI_CHECK(exp) mpi_safe_call(exp, __FILE__, __LINE__)

int MPISafeCall(int ret_code, const std::string &fname, int line);


#endif // UTILS_H
