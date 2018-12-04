/**
 *  @file mpi_eigen_helpers.h
 *  @brief Helpers for working with Eigen data structures over MPI.
 */
#ifndef HPSC_MPI_EIGEN_HELPERS_H
#define HPSC_MPI_EIGEN_HELPERS_H

#include <memory>

#include "common/eigen_helpers.h"
#include "common/mpi_helpers.h"

// TODO: method to bcast dense matrix too!

/**
 * @brief Broadcasts the given sparse Eigen matrix using MPI, starting from the given node.
 * @param A         The sparse matrix to broadcast.
 * @param sender    The index of the root node.
 */
void BroadcastEigenSparse(ESMatrix &A, int sender = 0);


/**
 * TODO(andreib): Document this properly once you finish coding.
 *  @brief Assembles [n_procs] input chunks of size [a x m] into a matrix of [m x a] in each processor.
 *  @param[in] in_chunk     [a x m] matrix owned locally by a processor.
 *  @param[out] out
 */
void AllToAllEigenDense(const EMatrix &in_chunk, EMatrix &out);


#endif //HPSC_MPI_EIGEN_HELPERS_H
