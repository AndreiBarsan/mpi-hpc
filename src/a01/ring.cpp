#include <chrono>
#include <string>
#include <vector>
#include <thread>

#include "gflags/gflags.h"
#include "mpi.h"

#include "src/common/utils.h"


using namespace std;

DEFINE_int32(iterations, 10, "The number of times to run each experiment.");


/// Performs the experiment from Assignment 1, Exercise 4.
template<typename T>
void RingCommunication(std::vector<T> &data, int n, bool group_before_transfer) {
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  MPI_Status istatus;
  int next = (local_id + 1);
  if (next >= n_procs) {
    next -= n_procs;
  }
  int prev = (local_id - 1) % n_procs;
  if (prev < 0) {
    prev += n_procs;
  }

  MPI_Request req;
  if (group_before_transfer) {
    // send and receive just once using a big buffer
    auto *send_buffer = new T[n];
    auto *recv_buffer = new T[n];
    for (int i = 0; i < n; i += 1) {
      send_buffer[i] = data[i * n];
    }

    // Send asynchronously to avoid deadlocks.
    MPI_Isend(send_buffer, n, MPIType<T>(), next, 0, MPI_COMM_WORLD, &req);
    MPI_Recv(recv_buffer, n, MPIType<T>(), prev, 0, MPI_COMM_WORLD, &istatus);
    MPI_Wait(&req, &istatus);

    for (int i = 0; i < n; i += 1) {
      data[i * n] = recv_buffer[i];
    }

    delete send_buffer;
    delete recv_buffer;
  }
  else {
    // many sends and recvs
    T send_buffer;
    T recv_buffer;
    for (int i = 0; i < n; i += 1) {
      send_buffer = data[i * n];
      // Deadlocks are basically impossible in this case, but we want to use async sends just so this code is
          // equivalent to the grouped one.
//      MPI_Isend(&send_buffer, 1, MPIType<T>(), next, 0, MPI_COMM_WORLD, &req);

      // No asnyc sends. No wait, so throughput is maximized.
      MPI_Send(&send_buffer, 1, MPIType<T>(), next, 0, MPI_COMM_WORLD);
      MPI_Recv(&recv_buffer, 1, MPIType<T>(), prev, 0, MPI_COMM_WORLD, &istatus);
//      MPI_Wait(&req, &istatus);

      data[i * n] = recv_buffer;
    }
  }

}

int RingExperiment(int argc, char **argv) {
  const unsigned int n_runs = FLAGS_iterations;
  const std::vector<int> ns = {1000, 2000, 4000, 8000, 16000};
  MPI_Init(&argc, &argv);
  int local_id = -1, n_procs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  // Make each node generate different data.
  srand(local_id);
  if (local_id == 0) {
    cout << "Will run for " << n_runs << " iterations." << endl;
  }

  for(int n : ns) {
    vector<chrono::duration<double>> times_grouped_s;
    vector<chrono::duration<double>> times_individ_s;
    vector<double> dummy_data_A;
    vector<double> dummy_data_B;

    for (int i = 0; i < n * n; ++i) {
      auto val = (double) rand() / (double) RAND_MAX;
      dummy_data_A.push_back(val);
      dummy_data_B.push_back(val);
    }

    for(unsigned int run_idx = 0; run_idx < n_runs; ++run_idx) {
      if (run_idx % 10 == 0) {
        cout << "Iteration " << run_idx << " for n = " << n << "." << endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      auto start_grouped = chrono::system_clock::now();
      RingCommunication(dummy_data_A, n, true);
      auto end_grouped = chrono::system_clock::now();

      MPI_Barrier(MPI_COMM_WORLD);
      auto start_individ = chrono::system_clock::now();
      RingCommunication(dummy_data_B, n, false);
      auto end_individ = chrono::system_clock::now();

      times_grouped_s.emplace_back(end_grouped - start_grouped);
      times_individ_s.emplace_back(end_individ - start_individ);

      // Add some sleeps every now and then just to make sure the run times are sampled over a longer period of time.
      if (run_idx % 5 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      if (run_idx % 10 == 0) {
        // Validate results every now and then
        for (int i = 0; i < dummy_data_A.size(); ++i) {
          if (dummy_data_A[i] != dummy_data_B[i]) {
            throw runtime_error("Inconsistent results between the two approaches!");
          }
        }
      }
    }

    string fpath_grouped = Format("../results/e04-grouped-n-%02d-%02d.csv", n, n_procs);
    string fpath_individ = Format("../results/e04-individ-n-%02d-%02d.csv", n, n_procs);
    WriteTimingResults(fpath_grouped, times_grouped_s);
    WriteTimingResults(fpath_individ, times_individ_s);
  }

  MPI_Finalize();
  return 0;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RingExperiment(argc, argv);
}
