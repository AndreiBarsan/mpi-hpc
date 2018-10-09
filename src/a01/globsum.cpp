#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <fstream>
#include <src/common/utils.h>
#include <thread>
#include <tuple>

#include "gflags/gflags.h"

#include "mpi.h"

static const int NO_TAG = 0;
static const unsigned int RANDOM_SEED = 1234;

DEFINE_bool(multiple_ops, false, "Whether to perform multiple operations instead of just the sum.");


template<typename T>
std::tuple<T, T, T, T> AllReduceMultiple(const std::vector<T> &data, bool manual_reduce) {
  using namespace std;
  const int op_count = 4;
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  T buf_send[op_count], buf_recv[op_count];

  // Only support processor counts which are PoT for now.
  auto d = static_cast<int>(log2(n_procs));
  if (1 << d != n_procs) {
    throw runtime_error("The number of MPI nodes must be a power of two.");
  }

  auto partition_size = static_cast<size_t>(ceil(data.size() / n_procs));
  size_t my_start = partition_size * local_id;

  T local_sum = 0;
  T local_prod = 1.0;
  T local_min = numeric_limits<T>::max();
  T local_max = numeric_limits<T>::min();
  for (size_t i = my_start; i < my_start + partition_size; ++i) {
    if (i < data.size()) {
      local_sum += data[i];
      local_prod *= data[i];
      local_min = min(local_min, data[i]);
      local_max = max(local_max, data[i]);
    }
  }

  T global_sum = local_sum;
  T global_prod = local_prod;
  T global_min = local_min;
  T global_max = local_max;
  if (manual_reduce) {
    for (int i = 0; i < d; ++i) {
      // If we're doing multiple operations, let's be smart about it: pack all the operands into one send/receive
      // pair to save resources.
      buf_send[0] = global_sum;
      buf_send[1] = global_prod;
      buf_send[2] = global_min;
      buf_send[3] = global_max;
      int dst = Flip((unsigned) i, (unsigned) local_id);
      MPI_Status istatus;
      MPI_Send(buf_send, op_count, MPIType<T>(), dst, 0, MPI_COMM_WORLD);
      MPI_Recv(buf_recv, op_count, MPIType<T>(), dst, 0, MPI_COMM_WORLD, &istatus);
      global_sum += buf_recv[0];
      global_prod *= buf_recv[1];
      global_min = min(global_min, buf_recv[2]);
      global_max = max(global_max, buf_recv[3]);
    }
  }
  else {
    buf_send[0] = global_sum;
    MPI_Allreduce(buf_send, buf_recv, 1, MPIType<T>(), MPI_SUM, MPI_COMM_WORLD);
    global_sum = buf_recv[0];

    buf_send[0] = global_prod;
    MPI_Allreduce(buf_send, buf_recv, 1, MPIType<T>(), MPI_PROD, MPI_COMM_WORLD);
    global_prod = buf_recv[0];

    buf_send[0] = global_min;
    MPI_Allreduce(buf_send, buf_recv, 1, MPIType<T>(), MPI_MIN, MPI_COMM_WORLD);
    global_min = buf_recv[0];

    buf_send[0] = global_max;
    MPI_Allreduce(buf_send, buf_recv, 1, MPIType<T>(), MPI_MAX, MPI_COMM_WORLD);
    global_max = buf_recv[0];
  }

  return make_tuple(global_sum, global_prod, global_min, global_max);
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
  auto partition_size = static_cast<size_t>(ceil(data.size() / n_procs));
  size_t my_start = partition_size * local_id;

  if (local_id == 0) {
    cout << "p = " << n_procs << " processors. Each processor will handle " << partition_size
         << " elements before performing the reduction." << endl;
  }

  T local_result = 0;
  for (size_t i = my_start; i < my_start + partition_size; ++i) {
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


template<int i, typename T>
void Check(std::tuple<T, T, T, T> manual, std::tuple<T, T, T, T> builtin) {
  using namespace std;
  T manual_res = get<i>(manual);
  T builtin_res = get<i>(builtin);
  if (fabs(manual_res - builtin_res) > 1e-6) {
    stringstream ss;
    ss << "Different results obtained with manual (" << manual_res << ") and built-in (" << builtin_res
       << ") versions of all-reduce!";
    throw runtime_error(ss.str()); // NOLINT
  }
}


int AllReduceBenchmark(int argc, char **argv) {
  using namespace std;

  const unsigned int n_runs = 250;
  const int n_samples = 1U << 26U;
  srand(RANDOM_SEED);
  MPI_Init(&argc, &argv);
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  vector<chrono::duration<double>> times_manual_s;
  vector<chrono::duration<double>> times_builtin_s;

  // Generate the dummy data once (in every processor, since we don't care about benchmarking the data sharing (yet)).
  // TODO(andreib): Use the modern C++ random library for much cleaner code.
  vector<double> dummy_data;
  for (int i = 0; i < n_samples; ++i) {
    dummy_data.push_back((double) rand() / (double) RAND_MAX);
  }

  if (FLAGS_multiple_ops) {
    for (unsigned int run_idx = 0; run_idx < n_runs; ++run_idx) {
      MPI_Barrier(MPI_COMM_WORLD);
      auto start_manual = chrono::system_clock::now();
      auto result_manual = AllReduceMultiple(dummy_data, true);
      auto end_manual = chrono::system_clock::now();

      MPI_Barrier(MPI_COMM_WORLD);
      auto start_builtin = chrono::system_clock::now();
      auto result_builtin = AllReduceMultiple(dummy_data, false);
      auto end_builtin = chrono::system_clock::now();

      if (local_id == 0) {
        Check<0>(result_manual, result_builtin);

        times_manual_s.emplace_back(end_manual - start_manual);
        times_builtin_s.emplace_back(end_builtin - start_builtin);
      }

      // Add some sleeps every now and then just to make sure the run times are sampled over a longer period of time.
      if (run_idx % 5 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (run_idx % 10 == 0 && local_id == 0) {
        cout << "Manual results: sum=" << get<0>(result_manual) << ", prod=" << get<1>(result_manual) << ", min="
            << get<2>(result_manual) << ", max=" << get<3>(result_manual) << endl;
        cout << "Built-in results: sum=" << get<0>(result_builtin) << ", prod=" << get<1>(result_builtin) << ", min="
             << get<2>(result_builtin) << ", max=" << get<3>(result_builtin) << endl;
      }
    }

//    if (local_id == 0) {
//      double res = accumulate(dummy_data.cbegin(), dummy_data.cend(), 0.0);
//      cout << "Single-thread result:" << res << endl;
//    }
  }
  else {
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

      // Add some sleeps every now and then just to make sure the run times are sampled over a longer period of time.
      if (run_idx % 5 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    if (local_id == 0) {
      double res = accumulate(dummy_data.cbegin(), dummy_data.cend(), 0.0);
      cout << "Single-thread result:" << res << endl;
    }
  }

  if (local_id == 0) {
    string label = FLAGS_multiple_ops ? "multiple" :  "sum";
    string fpath_manual = Format("results/manual-%s-%02d.csv", label.c_str(), n_procs);
    WriteTimingResults(fpath_manual, times_manual_s);
    string fpath_builtin = Format("results/builtin-%s-%02d.csv", label.c_str(), n_procs);
    WriteTimingResults(fpath_builtin, times_builtin_s);

    cout << "Wrote results." << endl;
  }

  MPI_Finalize();
  return 0;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return AllReduceBenchmark(argc, argv);
}
