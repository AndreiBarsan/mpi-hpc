#include "parallel_numerical.h"

#include <map>

#include "Eigen/LU"

// Known to be ALWAYS zero: B_i2, C_i2.
// TODO-LOW(andreib): Implement low-tri/up-tri matrix class to improve efficiency of certain multiplications even more.

Matrix<double> SolveParallel(BandMatrix<double> &A, Matrix<double> &b) {
  MPI_SETUP;
  uint32_t n = A.get_n();
  uint32_t q = static_cast<uint32_t>(n / n_procs);
  uint32_t beta = 2 * A.get_bandwidth() + 1;    // The effective bandwidth of the matrix.
  MPI_Request send_request;                     // Used in asynchronous sends.
  MPI_Status status;
  if (q <= 1) {
    throw runtime_error(Format("Too many processors for such a small system! You end up with a partition size of just"
                               " %uld (%uld equations, %d processors)", q, n, n_procs));
  }
  assert(q * n_procs == n);                     // Validate the assumption that n_procs perfectly divides n.

  // Note: the make_unique<T[]> support is a C++14 thing.
  auto send_buffer = make_unique<double[]>(A.get_n() * A.get_bandwidth() * 2);
  auto recv_buffer = make_unique<double[]>(A.get_n() * A.get_bandwidth() * 2);

  MASTER {
    // Node 0 is considered the master node. We assume it is the only node which originally has the full A and b,
    // even though in reality we compute A and b on each processor. This ensures that our 'SolveParallel' method is
    // appropriately generic.
    cout << "Running parallel solver on " << n_procs << " processors for " << n << "x" << n << " system." << endl;

    for(int i = 0; i < n_procs; ++i) {
      // Send all the data necessary for one processor in a single transfer to reduce overhead as much as possible.
      uint32_t count = A.write_raw_rows(i * q, (i + 1) * q, send_buffer.get(), 0);
      assert (count == (q * 3));

      count = b.write_raw_rows(i * q, (i + 1) * q, send_buffer.get(), count);
      assert (count == (q * 3) + q);

      // Do not send to yourself in a blocking manner, as it deadlocks on macOS (but not on Linux, it seems).
      if (i != 0) {
        MPI_Send(send_buffer.get(), count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      }
      else {
        for (uint32_t j = 0; j < count; ++j) {
          recv_buffer.get()[j] = send_buffer.get()[j];
        }
      }
    }
  }

  // Receive our chunk of A, plus the corresponding data from b.
  if (local_id != 0) {
    MPI_Recv(recv_buffer.get(), 3 * q + q, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
  }

  // We got our data, now let's assemble our matrices. We start with Ai1, which is still a banded tridiagonal matrix
  // of dimensions (q - b) x (q - b).
  vector<double> A_i_1_data;
  vector<double> A_i_4_data;
  for(uint32_t i = 0; i < 3 * (q - beta); ++i) {
    // First element pushed is not actually part of the matrix. That is fine, since in first block it's a padding
    // value from the original matrix, and in the following blocks they are elements actually contained in the C block.
    A_i_1_data.push_back(recv_buffer.get()[i]);
  }

  // In the simple tridiagonal spline interpolation case, A_i_2 and A_i_3 have just 1 element!
  // TODO-LOW(andreib): For future support for arbitrary bands, modify this part.
  Matrix<double> A_i_2(q - beta, beta, Zeros((q - beta) * beta));
  Matrix<double> A_i_3(beta, q - beta, Zeros(beta * (q - beta)));
  A_i_2(q - beta - 1, 0) = recv_buffer[3 * (q - beta) - 1];
  A_i_3(0, q - beta - 1) = recv_buffer[3 * (q - beta)];

  for(uint32_t i = 3 * (q - beta); i < 3 * q; ++i) {
    A_i_4_data.push_back(recv_buffer.get()[i]);
  }
  assert(A_i_4_data.size() == 9);

  Matrix<double> B_i_1(beta, q - beta, Zeros(beta * (q - beta)));
  // In the 1D quad spline case, this is always entirely zero!
  Matrix<double> B_i_2(beta, beta, Zeros(beta * beta));
  Matrix<double> C_i_1(q - beta, beta, Zeros((q - beta) * beta));
  // In the 1D quad spline case, this is always entirely zero!
  Matrix<double> C_i_2(beta, beta, Zeros(beta * beta));

  Matrix<double> b_i_1(q - beta, 1, Zeros(q - beta));
  Matrix<double> b_i_2(beta, 1, Zeros(beta));

  uint32_t offset = 3 * q;
  offset = b_i_1.set_from(recv_buffer, offset);
  offset = b_i_2.set_from(recv_buffer, offset);

  BandMatrix<double> A_i_1(static_cast<uint32_t>(q - beta), A_i_1_data);
  BandMatrix<double> A_i_4(static_cast<uint32_t>(beta), A_i_4_data);

  B_i_1(beta - 1, 0) = recv_buffer[3 * q - 1];
  if (local_id > 0) {
    // There is no C_1 block for first proc.
    C_i_1(0, beta - 1) = recv_buffer[0];
  }
  cout << "[PP2] " << local_id << ": data received; matrix assembly OK." << endl;

  // Step 1: LU factor in-place Ai1 in each processor i.
  BandedLUFactorization(A_i_1, true);

  // Step 2: Fwd/backsub equation 2 in each node.
  auto A_i_2_prime = SolveDecomposed(A_i_1, A_i_2);
  assert(A_i_2_prime.cols_ == beta);

  // Step 3: Fwd/backsub equation 3 in each node, except the first.
  Matrix<double> C_i_1_prime(C_i_1);
  if (local_id > 0) {
    C_i_1_prime = SolveDecomposed(A_i_1, C_i_1);
  }

  // Step 4: Fwd/backsub equation 4 in each node.
  auto b_i_1_prime = SolveDecomposed(A_i_1, b_i_1);
//  cout << "Debug PP2 in " << local_id << ", b_i_1' = " << b_i_1_prime << endl;
//  cout << "Debug PP2 in " << local_id << ", A_i_3  = " << A_i_3 << endl;

  // Step 4.5: Send C_i_1_prime back from all nodes except first. (Which doesn't have a proper C_i_1_prime anyway!)
  int C_i_1_prime_els = C_i_1_prime.rows_ * C_i_1_prime.cols_;
  if (local_id > 0) {
    int count = C_i_1_prime.write_raw(send_buffer.get(), 0);
    assert (count == C_i_1_prime_els);
    MPI_Send(send_buffer.get(), C_i_1_prime_els, MPI_DOUBLE, local_id - 1, 0, MPI_COMM_WORLD);
  }

  // Step 5: Receive C_i+1_1_prime and compute A_i_4_prime ill all nodes except the last.
  Matrix<double> C_i_plus_1_1_prime(C_i_1);
  Matrix<double> A_i_4_prime(beta, beta, Zeros(beta * beta));
  if (local_id < n_procs - 1) {
    MPI_Recv(recv_buffer.get(), C_i_1_prime_els, MPI_DOUBLE, local_id + 1, 0, MPI_COMM_WORLD, &status);
    C_i_plus_1_1_prime.set_from(recv_buffer.get());

    A_i_4_prime = A_i_4.get_dense() - A_i_3 * A_i_2_prime - B_i_1 * C_i_plus_1_1_prime;
  }
  else {
    // Perform a custom computation in the last node.
    A_i_4_prime = A_i_4.get_dense() - A_i_3 * A_i_2_prime;
  }

  // Step 5.5: Send A_i2 back from all but first proc. Recv A_i2 from all but last proc.
  if (local_id > 0) {
    uint32_t count = A_i_2_prime.write_raw(send_buffer.get(), 0);
    assert (count == A_i_2_prime.rows_ * A_i_2_prime.cols_);
    MPI_Send(send_buffer.get(), count, MPI_DOUBLE, local_id - 1, 0, MPI_COMM_WORLD);
  }

  // Step 6: Compute matrix mul/sub; Assume matrix is dense for now. (EXCEPT LAST PROC)
  Matrix<double> B_i_2_prime(B_i_2);
  if (local_id < n_procs - 1) {
    MPI_Recv(recv_buffer.get(), A_i_2_prime.rows_ * A_i_2_prime.cols_, MPI_DOUBLE, local_id + 1, 0, MPI_COMM_WORLD,
             &status);

    Matrix<double> A_i_plus_1_2_prime(A_i_2);
    A_i_plus_1_2_prime.set_from(recv_buffer.get());

    B_i_2_prime = B_i_2 - B_i_1 * A_i_plus_1_2_prime;
  }
  else {
    // Do nothing this time.
  }

  // Step 7: Compute the multiplication, except in the first processor (which obviously doesn't have a C block).
  Matrix<double> C_i_2_prime(C_i_2);
  if (local_id > 0) {
    C_i_2_prime = C_i_2 - A_i_3 * C_i_1_prime;
  }

  // Step 8: Do data transfer (send chunks of b_i,1' back) then appropriate operation branch.
  if (local_id > 0) {
    // Send b_i_1_prime back.
    uint32_t count = b_i_1_prime.write_raw(send_buffer.get(), 0);
    assert(count == b_i_1_prime.rows_);

    MPI_Send(send_buffer.get(), count, MPI_DOUBLE, local_id - 1, 0, MPI_COMM_WORLD);
  }

  Matrix<double> b_i_plus_1_1_prime(b_i_1_prime);
  Matrix<double> b_i_2_prime(b_i_2);
  if (local_id < n_procs - 1) {
    MPI_Recv(recv_buffer.get(), b_i_1_prime.rows_, MPI_DOUBLE, local_id + 1, 0, MPI_COMM_WORLD, &status);
    b_i_plus_1_1_prime.set_from(recv_buffer.get());

    b_i_2_prime = b_i_2 - A_i_3 * b_i_1_prime - B_i_1 * b_i_plus_1_1_prime;
  }
  else {
    b_i_2_prime = b_i_2 - A_i_3 * b_i_1_prime;
  }

  // Step 9: Send bits back to node 0 to form the reduced system, which should now be much smaller!
  // This could also be implemented by doing an all-to-all broadcast to assemble the reduced system in every node and
  // solve it in every node at the same time, avoiding to do communication for step 10. This method saves power,
  // since it only solves the reduced system on one node, but induces a little bit of extra latency because of the
  // addtiional communication. Either method should be correct, since perfect efficiency is not the main goal of this
  // assignment.
  uint32_t count = 0;
  if (local_id > 0) {
    count = C_i_2_prime.write_raw(send_buffer.get(), count);
    assert(count == C_i_2_prime.rows_ * C_i_2_prime.cols_);
  }

  uint32_t old_count = count;
  count = A_i_4_prime.write_raw(send_buffer.get(), count);
  assert(count - old_count == A_i_4_prime.rows_ * A_i_4_prime.cols_);

  if (local_id < n_procs - 1) {
    old_count = count;
    count = B_i_2_prime.write_raw(send_buffer.get(), count);
    assert(count - old_count == B_i_2_prime.rows_ * B_i_2_prime.cols_);
  }

  old_count = count;
  count = b_i_2_prime.write_raw(send_buffer.get(), count);
  assert(count - old_count == b_i_2_prime.rows_);

//  cout << "Will make primary send, count = " << count << "." << endl;
  MPI_Isend(send_buffer.get(), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &send_request);

  MASTER {
    // TODO(andreib): Store reduced A as banded matrix, bandwidth = 2 * beta.
    Matrix<double> reduced_A(n_procs * beta, n_procs * beta, Zeros(n_procs * beta * n_procs * beta));
    Matrix<double> reduced_b(n_procs * beta, 1, Zeros(n_procs * beta));

    for (int i = 0; i < n_procs; ++i) {
      count = A_i_4_prime.rows_ * A_i_4_prime.cols_ + b_i_2_prime.rows_;
      count += C_i_2_prime.rows_ * C_i_2_prime.cols_;
      if (i != 0 && i != n_procs - 1) {
        count += C_i_2_prime.rows_ * C_i_2_prime.cols_;
      }

      MPI_Recv(recv_buffer.get(), count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
      uint32_t cur_offset = 0;

      if(i > 0) {
        cur_offset = C_i_2_prime.set_from(recv_buffer.get(), cur_offset);
      }

      cur_offset = A_i_4_prime.set_from(recv_buffer.get(), cur_offset);
      if (i < n_procs - 1) {
        cur_offset = B_i_2_prime.set_from(recv_buffer.get(), cur_offset);
      }

      cur_offset = b_i_2_prime.set_from(recv_buffer.get(), cur_offset);
      assert(cur_offset == count);

      // The reduced system matrix consists of (beta x beta)-sized blocks.
      for(uint32_t j = 0; j < beta; ++j) {
        for(uint32_t k = 0; k < beta; ++k) {

          if (i > 0) {
            reduced_A(i * beta + j, (i - 1) * beta + k) = C_i_2_prime(j, k);
          }

          reduced_A(i * beta + j, i * beta + k) = A_i_4_prime(j, k);

          if (i < n_procs - 1) {
            reduced_A(i * beta + j, (i + 1) * beta + k) = B_i_2_prime(j, k);
          }
        }
      }

      for(uint32_t j = 0; j < beta; ++j) {
        reduced_b(i * beta + j, 0) = b_i_2_prime(j, 0);
      }
    }

//    cout << "PP2 reduced system:\n" << reduced_A << "\n PP2 reduced b:" << reduced_b << endl;
//    cout << "Original was: \n" << A << endl;

    vector<double> band_data;
    band_data.push_back(0.0);
    for(uint32_t j = 0; j < n_procs * beta; ++j) {
      if(j > 0) {
        band_data.push_back(reduced_A(j, j-1));
      }
      band_data.push_back(reduced_A(j, j));
      if (j < n_procs * beta - 1) {
        band_data.push_back(reduced_A(j, j + 1));
      }
    }
    band_data.push_back(0.0);

    BandMatrix<double> reduced_A_banded(reduced_A.rows_, band_data);
//    cout << reduced_A_banded << endl;

    // Solve reduced system on the MASTER node.
    auto x_i_2_all = SolveSerial(reduced_A_banded, reduced_b, true);

//    cout << "[PP2] Part of solution: " << x_i_2_all << endl;

    // Send x_2 chunks to all nodes so they can compute the x_1 chunks.
    for(int i = 0; i < n_procs; ++i) {
      for(uint32_t j = 0; j < beta; ++j) {
        send_buffer[j] = x_i_2_all(i * beta + j, 0);
      }
      MPI_Isend(send_buffer.get(), beta, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &send_request);
    }
  }

  MPI_Recv(recv_buffer.get(), beta, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
  Matrix<double> x_i_2(beta, 1UL, Zeros(beta));
  x_i_2.set_from(recv_buffer, 0);

  // Step 10: Compute missing chunks from x (xi1 for all but first proc, and x11 specifically).
  if (local_id < n_procs - 1) {
    // Send x_2 forward.
    x_i_2.write_raw(send_buffer, 0);
    MPI_Send(send_buffer.get(), beta, MPI_DOUBLE, local_id + 1, 0, MPI_COMM_WORLD);
  }

  Matrix<double> x_i_1(q - beta, 1UL, Zeros(q - beta));
  if (local_id > 0) {
    MPI_Recv(recv_buffer.get(), beta, MPI_DOUBLE, local_id - 1, 0, MPI_COMM_WORLD, &status);
    Matrix<double> x_i_minus_1_2(x_i_2);
    x_i_minus_1_2.set_from(recv_buffer, 0);
//    cout << "Debug PP2, i am " << local_id << " and the prev x_i_2 is " << x_i_minus_1_2 << endl;

    x_i_1 = b_i_1_prime - A_i_2_prime * x_i_2 - C_i_1_prime * x_i_minus_1_2;
  }
  else {
    x_i_1 = b_i_1_prime - A_i_2_prime * x_i_2;
  }

//  cout << "Exchanged x_2's OK and computed x_1's." << endl;
//  cout << "Debug PP2, " << local_id << ": x_1 = " << x_i_1 << endl;

  // Ok, now send all x_1's to the master node, 0.
  count = x_i_1.write_raw(send_buffer, 0);
  x_i_2.write_raw(send_buffer, count);
  MPI_Isend(send_buffer.get(), q, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &send_request);

  Matrix<double> x_all(n, 1, Zeros(n));
  MASTER {
    for(int i = 0; i < n_procs; ++i) {
      MPI_Recv(recv_buffer.get(), q, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
      int idx;

      for(uint32_t j = 0; j < q; ++j) {
        idx = i * q + j;
        x_all(idx, 0) = recv_buffer[j];
      }
    }

    x_all.write_raw(send_buffer, 0);
  }

  // Broadcast the final solution to all nodes so it can be returned.
  MPI_Bcast(send_buffer.get(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  x_all.set_from(send_buffer, 0);
  return x_all;
}

#ifdef DEBUG_WITH_EIGEN
Matrix<double> SolveParallelDebug(const BandMatrix<double> &A_raw, const Matrix<double> &b_raw) {
  using namespace Eigen;

  EMatrix A = ToEigen(A_raw);
  EMatrix b = ToEigen(b_raw);

  cout << A << endl << A_raw << endl;
  cout << b << endl << b_raw << endl;

  int n_procs = 4;
  map<int, EMatrix> A_1;
  map<int, EMatrix> A_2;
  map<int, EMatrix> A_3;
  map<int, EMatrix> A_4;
  map<int, EMatrix> B_1;
  map<int, EMatrix> B_2;
  map<int, EMatrix> C_1;
  map<int, EMatrix> C_2;
  map<int, EMatrix> b_1;
  map<int, EMatrix> b_2;
  map<int, EMatrix> x_1;
  map<int, EMatrix> x_2;

  int n = A.rows();
  int q = n / n_procs;
  int beta = 3;  // hardcoded for debug

  for(int i = 0; i < n_procs; ++i) {
    A_1[i] = EMatrix(q - beta, q - beta);
    A_2[i] = EMatrix(q - beta, beta);
    A_3[i] = EMatrix(beta, q - beta);
    A_4[i] = EMatrix(beta, beta);
    B_1[i] = EMatrix(beta, q - beta);
    B_2[i] = EMatrix::Zero(beta, beta);
    C_1[i] = EMatrix(q - beta, beta);
    C_2[i] = EMatrix::Zero(beta, beta);
    b_1[i] = EMatrix(q - beta, 1);
    b_2[i] = EMatrix(beta, 1);
    x_1[i] = EMatrix::Zero(q - beta, 1);
    x_2[i] = EMatrix::Zero(beta, 1);

    cout << A_1[i].rows() << ", " << A_1[i].cols() << endl;
    for (int j = 0; j < q - beta; ++j) {
      for (int k = 0; k < q - beta; ++k) {
        A_1[i](j, k) = A(i * q + j, i * q + k);
      }
    }

    for (int j = 0; j < q - beta; ++j) {
      for (int k = 0; k < beta; ++k) {
        A_2[i](j, k) = A(i * q + j, i * q + (q - beta) + k);
      }
    }

    for (int j = 0; j < beta; ++j) {
      for (int k = 0; k < q - beta; ++k) {
        A_3[i](j, k) = A(i * q + (q - beta) + j, i * q + k);
      }
    }

    for (int j = 0; j < beta; ++j) {
      for (int k = 0; k < beta; ++k) {
        A_4[i](j, k) = A(i * q + (q - beta) + j, i * q + (q - beta) + k);
      }
    }

    if (i < n_procs - 1) {    // Do not set B_1 in the last block.
      for (int j = 0; j < beta; ++j) {
        for (int k = 0; k < q - beta; ++k) {
          B_1[i](j, k) = A(i * q + (q - beta) + j, (i + 1) * q + k);
        }
      }
    }
    // B_2 is always full of zeros

    if (i > 0) {  // Do not set C_1 in the first block.
      for (int j = 0; j < q - beta; ++j) {
        for (int k = 0; k < beta; ++k) {
          C_1[i](j, k) = A(i * q + j, (i - 1) * q + (q - beta) + k);
        }
      }
    }
    // C_2 is always full of zeros

    for (int j = 0; j < q - beta; ++j) {
      b_1[i](j, 0) = b(i * q + j);
    }
    for (int j = 0; j < beta; ++j) {
      b_2[i](j, 0) = b(i * q + q - beta + j);
    }

//    cout << A << endl << b << endl;
//    cout << endl << endl << "Stuff in \"processor\" " << i << ":" << endl;
//    cout << "A_1:" << endl << A_1[i] << endl;
//    cout << "A_2:" << endl << A_2[i] << endl;
//    cout << "A_3:" << endl << A_3[i] << endl;
//    cout << "A_4:" << endl << A_4[i] << endl;
//    if (i < n_procs - 1) {
//      cout << "B_1:" << endl << B_1[i] << endl;
//      cout << "B_2:" << endl << B_2[i] << endl;
//    } else {
//      cout << "No B's in last block." << endl;
//    }
//    if (i > 0) {
//      cout << "C_1:" << endl << C_1[i] << endl;
//      cout << "C_2:" << endl << C_2[i] << endl;
//    } else {
//      cout << "No C's in first block." << endl;
//    }

    auto A_1_factored = A_1[i].fullPivLu();   // Eigen doesn't seem to have non-pivoting LU
    auto aux = A_1_factored.solve(A_2[i]);
    A_2[i] = aux;

    if (i > 0) {
      auto aux2 = A_1_factored.solve(C_1[i]);
      C_1[i] = aux2;
    }

    auto aux3 = A_1_factored.solve(b_1[i]);
    b_1[i] = aux3;
  }

  for(int i = 0; i < n_procs; ++i) {
    // Step 5
    if (i < n_procs - 1) {
      A_4[i] = A_4[i] - A_3[i] * A_2[i] - B_1[i] * C_1[i + 1];
    } else {
      A_4[i] = A_4[i] - A_3[i] * A_2[i];
    }
  }

  for(int i = 0; i < n_procs; ++i) {
    // Step 6
    if (i < n_procs - 1) {  // since no block B in last processor
      B_2[i] = B_2[i] - B_1[i] * A_2[i + 1];
    }
  }

  for(int i = 0; i < n_procs; ++i) {
    // Step 7
    if (i > 0) {    // since no block C in first processor
      C_2[i] = C_2[i] - A_3[i] * C_1[i];
    }
//    cout << "Step 7" << endl;
  }

  for(int i = 0; i < n_procs; ++i) {
    // Step 8
    if (i < n_procs - 1) {
      b_2[i] = b_2[i] - A_3[i] * b_1[i] - B_1[i] * b_1[i + 1];
    } else {
      // Drop last term since no block B in last processor
      b_2[i] = b_2[i] - A_3[i] * b_1[i];
    }
  }

  // Step 9: We now have the reduced system!
  EMatrix A_red = EMatrix::Zero(n_procs * beta, n_procs * beta);
  EMatrix b_red = EMatrix::Zero(n_procs * beta, 1);
  for(int i = 0; i < n_procs; ++i) {
    for (int j = 0; j < beta; ++j) {
      for (int k = 0; k < beta; ++k) {
        A_red(i * beta + j, i * beta + k) = A_4[i](j, k);
      }
    }

    if (i < n_procs - 1) {    // Do not use B_2 in the last block.
      for (int j = 0; j < beta; ++j) {
        for (int k = 0; k < beta; ++k) {
          A_red(i * beta + j, (i + 1) * beta + k) = B_2[i](j, k);
        }
      }
    }

    if (i > 0) {  // Do not use C_2 in the first block.
      for (int j = 0; j < beta; ++j) {
        for (int k = 0; k < beta; ++k) {
          A_red(i * beta + j, (i - 1) * beta + k) = B_2[i](j, k);
        }
      }
    }

    for(int j = 0; j < beta; ++j) {
      b_red(i * beta + j, 0) = b_2[i](j);
    }
  }

  cout << "[Eig] We reached the end. Reduced system:" << endl;
  cout << "[Eig] " << A_red << endl;
  cout << "[Eig] " << b_red << endl;

  EMatrix x_all_2 = A_red.fullPivLu().solve(b_red);
  cout << "[Eig] Solution to reduced system (part of full solution):" << endl;
//  cout << x_all_2.rows() << ", " << x_all_2.cols() << endl;
  cout << x_all_2 << endl;
  cout << endl << endl;

  EMatrix x_all = EMatrix::Zero(n, 1);

  // Step 10 to compute full solution and put everything together.
  for(int i = 0; i < n_procs; ++i) {
    for(int j = 0; j < beta; ++j) {
      x_2[i](j) = x_all_2(i*beta + j);
      x_all(i * q + (q - beta) + j) = x_2[i](j);
    }

    if (i > 0) {
      x_1[i] = b_1[i] - A_2[i] * x_2[i] - C_1[i] * x_2[i - 1];
    }
    else {
      x_1[i] = b_1[i] - A_2[i] * x_2[i];
    }

    cout << "Debug Eig, " << i  << ", x_1 = " << x_1[i] << endl;

    for(int j = 0; j < q - beta; ++j) {
      x_all(i * q + j) = x_1[i](j);
    }
  }

  return ToMatrix(x_all);
}
#endif
