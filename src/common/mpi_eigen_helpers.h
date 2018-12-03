/**
 *  @file mpi_eigen_helpers.h
 *  @brief Helpers for working with Eigen data structures over MPI.
 */
#ifndef HPSC_MPI_EIGEN_HELPERS_H
#define HPSC_MPI_EIGEN_HELPERS_H

#include "common/eigen_helpers.h"
#include "common/mpi_helpers.h"

// TODO: method to bcast dense matrix too!

/// Broadcasts the given sparse Eigen matrix using MPI, starting from the given node.
/// \param A        The sparse matrix to broadcast.
/// \param sender   The index of the root node.
void BroadcastEigenSparse(ESMatrix &A, int sender = 0);


#endif //HPSC_MPI_EIGEN_HELPERS_H
