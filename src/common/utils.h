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

bool EndsWith(const std::string &value, const std::string &ending);

std::string Format(const std::string& fmt, ...);

std::string Type2Str(int type);

std::string GetDate();



#endif // UTILS_H
