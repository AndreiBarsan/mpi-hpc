/**
 *  @file mpi_eigen_helpers_tests.cpp
 *  @brief Some simple manual tests for communicating Eigen matrices over MPI.
 *
 *  These are manual tests because I can't really be bothered to check how to run GTest tests with MPI. (Ant this is
 *  a course projects so time is limited.)
 */

#include <chrono>
#include <iostream>
#include <thread>

#include "common/mpi_helpers.h"
#include "common/mpi_eigen_helpers.h"

using namespace std;


ESMatrix get_sample_matrix(int rows, int cols) {
  ESMatrix mat;
  mat.resize(rows, cols);
  mat.insert(0, 0) = 19;
  for (int i = 0; i < rows; i += 2) {
    for (int j = 3; j < cols; j += 30) {
      mat.insert(i, j - 2) = 42.0;
      mat.insert(i, j) = 13.0;
    }
  }
  return mat;
}

void test_bcast_sparse_fixture(int rows, int cols, int sender) {
  MPI_SETUP;
  if (n_procs < 4) {
    throw runtime_error("Please run the test with MPI and at least 4 nodes.");
  }

  ESMatrix mat;
  if (local_id == sender) {
    mat = get_sample_matrix(rows, cols);
    cout << "Doing test matrix broadcast [" << rows << " x " << cols << "] to all our " << n_procs
         << " processors. Sender: " << sender << "\n";
  }
  // This is very important if we want to send this matrix over MPI!
  mat.makeCompressed();

  BroadcastEigenSparse(mat, sender);
  ESMatrix expected = get_sample_matrix(rows, cols);
  ESMatrix &actual = mat;
  assert(expected.isApprox(actual));
}

void test_bcast_sparse_square() {
  MPI_SETUP;
  int szs[] = {1, 2, 3, 4, 5, 8, 64, 1024};
  int senders[] = {0, 1, 2, 3, n_procs - 1};
  for(int sender : senders) {
    for (int sz : szs) {
      test_bcast_sparse_fixture(sz, sz, sender);
    }
  }
}

void test_bcast_sparse_nonsquare() {
  MPI_SETUP;
  int rows[] = {1, 2, 3, 4, 5, 8, 64, 1024};
  int cols[] = {1, 2, 3, 4, 5, 8, 64, 1024};
  int senders[] = {0, 1, 2, 3, n_procs - 1};

  for(int sender : senders) {
    for (int row : rows) {
      for (int col : cols) {
        test_bcast_sparse_fixture(row, col, sender);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }

}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_SETUP;
  test_bcast_sparse_square();
  MASTER {
    cout << "Square matrix tests OK." << endl;
  }
  test_bcast_sparse_nonsquare();
  MASTER {
    cout << "Non-square matrix tests OK." << endl;
  }
  MPI_Finalize();
  return 0;
}
