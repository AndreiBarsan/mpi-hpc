#include "utils.h"

template<typename Out>
void Split(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> Split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  Split(s, delim, back_inserter(elems));
  return elems;
}

bool PathExists(const std::string &path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

bool IsDir(const std::string &path) {
  struct stat info;

  if (stat(path.c_str(), &info) != 0) {
    return false;
  }

  return (info.st_mode & S_IFDIR) != 0;
}

int MPISafeCall(int ret_code, const std::string &fname, int line) {
  if (MPI_SUCCESS == ret_code) {
    return ret_code;
  } else {
    // There was an error: throw an exception with a meaningful error message.
    std::stringstream ss;
    char err_msg[1024];
    int len;
    MPI_Error_string(ret_code, err_msg, &len);
    ss << "MPI call failed in file " << fname << " on line " << line << " with error code " << ret_code
       << "(" << err_msg << ").";
    throw std::runtime_error(ss.str());
  }
}
