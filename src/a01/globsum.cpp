#include <iostream>
#include <sstream>

#include "mpi.h"

#define MPI_CHECK(exp) mpi_safe_call(exp, __FILE__, __LINE__)

// TODO(andreib): use gflags!

static const int NO_TAG = 0;

int mpi_safe_call(int ret_code, const std::string &fname, int line) {
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

/**
 * Hello world in Open MPI.
 */
void hello(int argc, char **argv) {
  using namespace std;
  int rank = -1, size = -1;
  MPI_CHECK(MPI_Init(&argc, &argv));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  cout << "I'm process " << rank << " of " << size << "." << endl;

  MPI_CHECK(MPI_Finalize());
}

int simple_communication(int argc, char **argv) {
  using namespace std;
  constexpr int N = 10;
  float vsend[N], vrecv[N];

  int local_id = -1, n_procs = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  if (0 == local_id) {
    cout << "Simple communication example..." << endl;
  }
  char proc_name[1024];
  int proc_name_len;
  MPI_Get_processor_name(proc_name, &proc_name_len);
  cout << "Processor " << local_id << "/" << n_procs << "." << endl;

  // Generate some dummy data to exchange
  for (int i = 0; i < N; ++i) {
    vsend[i] = local_id * N + i;
  }

  // Send data to right neighbor and get data from left neighbor, wrapping around.
  int l_neighbor_id = (local_id - 1) % n_procs;
  int r_neighbor_id =(local_id + 1) % n_procs;

  MPI_Status istatus;
  MPI_Send(vsend, N, MPI_FLOAT, r_neighbor_id, NO_TAG, MPI_COMM_WORLD);
  MPI_Recv(vrecv, N, MPI_FLOAT, l_neighbor_id, NO_TAG, MPI_COMM_WORLD, &istatus);

  cout << "Received in " << local_id << " OK!" << endl;
  stringstream ss;
  ss << "My numbers (" << local_id << "): ";
  for (float i : vsend) {
    ss << i << " ";
  }
  ss << "| Received numbers: ";
  for (float i : vrecv) {
    ss << i << " ";
  }
  cout << ss.str() << endl;


  MPI_Finalize();
  return 0;
}

int main(int argc, char **argv) {
  return simple_communication(argc, argv);
}