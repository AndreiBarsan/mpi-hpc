#include <iostream>
#include <sstream>

#include "mpi.h"

#define MPI_CHECK(exp) mpi_safe_call(exp, __FILE__, __LINE__)

int mpi_safe_call(int ret_code, const std::string &fname, int line) {
    if (MPI_SUCCESS == ret_code) {
        return ret_code;
    }
    else {
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

int main(int argc, char **argv) {
    using namespace std;
    // TODO(andreib): use gflags!
    int rank = -1, size = -1;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    cout << "I'm process " << rank << " of " << size << "." << endl;

    MPI_CHECK(MPI_Finalize());
    return 0;
}