#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>

#include "mpi.h"

// TODO(andreib): use gflags!
static const int NO_TAG = 0;
static const unsigned int RANDOM_SEED = 1234;

// A little bit of template hacking..er.. magic to tame MPI!
template<typename T>
MPI_Datatype MPIType();

template<>
MPI_Datatype MPIType<double>() {
  return MPI_DOUBLE;
};

template<>
MPI_Datatype MPIType<float>() {
  return MPI_FLOAT;
};

int Flip(unsigned int i, unsigned int n) {
  unsigned int mask = 1 << i;
  if (n & mask) {
    // the bit is set: un-set it
    return n & (~mask);
  } else {
    return n | mask;
  }
}

template<typename T>
int AllReduceSumManual(const std::vector<T> &data) {
  using namespace std;
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  T buf_send, buf_recv;

  // assume proc count is PoT for now
  int d = static_cast<int>(log2(n_procs));
  cout << "d = " << d << endl;

  int partition_size = static_cast<int>(ceil(data.size() / n_procs));
  int my_start = partition_size * local_id;

  T local_result = 0;
  for (int i = my_start; i < my_start + partition_size; ++i) {
    if (i < data.size()) {
      // Ensure we deal with cases where the element count is a non-PoT.
      local_result += data[i];
    }
  }

  T global_result = local_result;
  for (int i = 0; i < d; ++i) {
    buf_send = global_result;
    int dst = Flip(i, local_id);
    MPI_Status istatus;
    MPI_Send(&buf_send, 1, MPIType<T>(), dst, 0, MPI_COMM_WORLD);
    MPI_Recv(&buf_recv, 1, MPIType<T>(), dst, 0, MPI_COMM_WORLD, &istatus);
    global_result += buf_recv;
  }
  cout << "Global result [np = " << local_id << "]:" << global_result << endl;

  return 0;
}

template<typename T>
int AllReduceSumBuiltin(const std::vector<T> &data) {
  // TODO implement
  return 0;
}

int AllReduceBenchmark(int argc, char **argv) {
  using namespace std;
  const int n_samples = 1 << 20;
  srand(RANDOM_SEED);
  MPI_Init(&argc, &argv);

  // TODO(andreib): Use the C++ random library for much cleaner code.
  vector<double> dummy_data;
  for (int i = 0; i < n_samples; ++i) {
    dummy_data.push_back((double) rand() / (double) RAND_MAX);
  }

  AllReduceSumManual(dummy_data);
  AllReduceSumBuiltin(dummy_data);

  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  if (local_id == 0) {
    double res = accumulate(dummy_data.cbegin(), dummy_data.cend(), 0.0);
    cout << "Single-thread result:" << res << endl;
  }

  MPI_Finalize();
  return 0;
}

int SimpleCommunication(int argc, char **argv) {
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
    vsend[i] = (local_id + 1) * N + i;
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
//  return SimpleCommunication(argc, argv);
  return AllReduceBenchmark(argc, argv);
}