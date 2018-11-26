#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <fstream>

#include <Eigen/Core>
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

/// Returns a date string such as 2018-01-31.
std::string GetDate();

/// Returns a string with the current working directory.
std::string GetCWD();

bool IsPowerOfTwo(unsigned int n);

// A little bit of template hacking..er.. magic to tame MPI!
// These helpers allow us to use MPI types from generic functions (provided the generic type is a basic type like
// int/float/etc. that's supported by MPI, of course).
template<typename T>
MPI_Datatype MPIType();

template<>
MPI_Datatype MPIType<double>();

template<>
MPI_Datatype MPIType<float>();

int Flip(unsigned int i, unsigned int n);

/// Writes the given timing data as a simple CSV.
void WriteTimingResults(std::string &fpath, const std::vector<std::chrono::duration<double>>& times_s);

/// Identical functionality to the 'linspace' function from numpy.
std::vector<double> Linspace(double a, double b, int n);

/// Returns the cartesian product of the sets x and y in a (xy) x 2 array.
/// Sort of like numpy.meshgrid in Python.
Eigen::MatrixX2d MeshGrid(const Eigen::ArrayXd &x, const Eigen::ArrayXd &y);


#endif // UTILS_H
