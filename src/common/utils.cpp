#include "utils.h"

#include <memory>
#include <iomanip>

using namespace std;

template<typename Out>
void Split(const string &s, char delim, Out result) {
  stringstream ss(s);
  string item;
  while (getline(ss, item, delim)) {
    *(result++) = item;
  }
}

vector<string> Split(const string &s, char delim) {
  vector<string> elems;
  Split(s, delim, back_inserter(elems));
  return elems;
}

bool PathExists(const string &path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

bool IsDir(const string &path) {
  struct stat info;

  if (stat(path.c_str(), &info) != 0) {
    return false;
  }

  return (info.st_mode & S_IFDIR) != 0;
}

bool EndsWith(const string &value, const string &ending) {
  if (ending.size() > value.size()) {
    return false;
  }

  return equal(ending.rbegin(), ending.rend(), value.rbegin());
}

string Format(const string &fmt, ...) {
  // TODO-LOW(andrei): Use varadic templates to implement in a typesafe, string-friendly manner.
  // Keeps track of the resulting string size.
  size_t out_size = fmt.size() * 2;
  unique_ptr<char[]> formatted;
  va_list ap;
  while (true) {
    formatted.reset(new char[out_size]);
    strcpy(&formatted[0], fmt.c_str());
    va_start(ap, fmt);
    int final_n = vsnprintf(&formatted[0], out_size, fmt.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || static_cast<size_t>(final_n) >= out_size) {
      int size_update = final_n - static_cast<int>(out_size) + 1;
      out_size += abs(size_update);
    }
    else {
      break;
    }
  }

  return string(formatted.get());
}

string GetDate() {
  time_t rawtime;
  tm * timeinfo;
  char today_s[200];
  time (&rawtime);
  timeinfo = localtime (&rawtime);
  strftime(today_s, 200, "%Y-%m-%d",timeinfo);

  return std::string(today_s);
}

int Flip(unsigned int i, unsigned int n) {
  unsigned int mask = 1 << i;
  if (n & mask) {
    // the bit is set: un-set it
    return n & (~mask);
  } else {
    return n | mask;
  }
}

template<>
MPI_Datatype MPIType<float>() {
  return MPI_FLOAT;
}

bool is_power_of_two(unsigned int n) {
  return n!= 0 && !((n - 1) & n);
}

void WriteTimingResults(std::string &fpath, const std::vector<std::chrono::duration<double>> &times_s) {
  std::ofstream file(fpath);
  if (!file) {
    throw std::runtime_error(Format("Could not write outputs to file %s.", fpath.c_str()));
  }
  file << "run, time_s" << std::endl;
  int i = 0;
  for(const auto& time_s : times_s) {
    file << i++ << "," << setprecision(12) << time_s.count() << std::endl;
  }
}

template<>
MPI_Datatype MPIType<double>() {
  return MPI_DOUBLE;
}

