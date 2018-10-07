#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <fstream>
#include <src/common/utils.h>

#include "mpi.h"

// TODO(andreib): use gflags
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
int AllReduceSum(const std::vector<T> &data, bool manual_reduce) {
  using namespace std;
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  T buf_send, buf_recv;

  // Only support processor counts which are PoT for now.
  auto d = static_cast<int>(log2(n_procs));
  if (1 << d != n_procs) {
    throw runtime_error("The number of MPI nodes must be a power of two.");
  }

  // From lecture notes, p. 1-37: "If p < n, some processors are assigned more than one component each. Each
  // processor sums-up its own components sequentially, then proceeds" (in the parallel manner).
  auto partition_size = static_cast<int>(ceil(data.size() / n_procs));
  int my_start = partition_size * local_id;

  if (local_id == 0) {
    cout << "p = " << n_procs << " processors. Each processor will handle " << partition_size
         << " elements before performing the reduction." << endl;
  }

  T local_result = 0;
  for (int i = my_start; i < my_start + partition_size; ++i) {
    if (i < data.size()) {
      // Ensure we deal with cases where the element count is a non-PoT.
      local_result += data[i];
    }
  }

  // Even though a regular reduction would be enough, in this exercise we perform an all-reduce, such that at the end
  // EVERY node has the global result.
  T global_result = local_result;
  if (manual_reduce) {
    for (int i = 0; i < d; ++i) {
      buf_send = global_result;
      int dst = Flip((unsigned) i, (unsigned) local_id);
      MPI_Status istatus;
      MPI_Send(&buf_send, 1, MPIType<T>(), dst, 0, MPI_COMM_WORLD);
      MPI_Recv(&buf_recv, 1, MPIType<T>(), dst, 0, MPI_COMM_WORLD, &istatus);
      global_result += buf_recv;
    }
  }
  else {
    buf_send = global_result;
    MPI_Allreduce(&buf_send, &buf_recv, 1, MPIType<T>(), MPI_SUM, MPI_COMM_WORLD);
    global_result = buf_recv;
  }

//  cout << "Global result [p = " << local_id << "]: " << global_result << endl;
  return global_result;
}

void WriteTimingResults(std::ofstream &file, const std::vector<std::chrono::duration<double>>& times_s) {
  file << "run, time_s" << std::endl;
  int i = 0;
  for(const auto& time_s : times_s) {
    file << i++ << ", " << time_s.count() << std::endl;
  }
}

int AllReduceBenchmark(int argc, char **argv) {
  using namespace std;

  const unsigned int n_runs = 50;
  const int n_samples = 1U << 24U;
  srand(RANDOM_SEED);
  MPI_Init(&argc, &argv);
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  vector<chrono::duration<double>> times_manual_s;
  vector<chrono::duration<double>> times_builtin_s;

  // Generate the dummy data once (in every processor, since we don't care about benchmarking the data sharing (yet)).
  // TODO(andreib): Use the C++ random library for much cleaner code.
  vector<double> dummy_data;
  for (int i = 0; i < n_samples; ++i) {
    dummy_data.push_back((double) rand() / (double) RAND_MAX);
  }

  for (unsigned int run_idx = 0; run_idx < n_runs; ++run_idx) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_manual = chrono::system_clock::now();
    double result_manual = AllReduceSum(dummy_data, true);
    auto end_manual = chrono::system_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_builtin = chrono::system_clock::now();
    double result_builtin = AllReduceSum(dummy_data, false);
    auto end_builtin = chrono::system_clock::now();

    if (local_id == 0) {
      if (fabs(result_manual - result_builtin) > 1e-6) {
        stringstream ss;
        ss << "Different results obtained with manual (" << result_manual << ") and built-in (" << result_builtin
           << ") versions of all-reduce!";
        throw runtime_error(ss.str()); // NOLINT
      }

      times_manual_s.emplace_back(end_manual - start_manual);
      times_builtin_s.emplace_back(end_builtin - start_builtin);
    }
  }

  if (local_id == 0) {
    double res = accumulate(dummy_data.cbegin(), dummy_data.cend(), 0.0);
    cout << "Single-thread result:" << res << endl;

    ofstream f_manual(Format("../results/manual-%02d.csv", n_procs));
    WriteTimingResults(f_manual, times_manual_s);

    ofstream f_builtin(Format("../results/builtin-%02d.csv", n_procs));
    WriteTimingResults(f_manual, times_builtin_s);

    cout << "Wrote results." << endl;
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
