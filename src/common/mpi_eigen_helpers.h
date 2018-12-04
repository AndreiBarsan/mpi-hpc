/**
 *  @file mpi_eigen_helpers.h
 *  @brief Helpers for working with Eigen data structures over MPI.
 */
#ifndef HPSC_MPI_EIGEN_HELPERS_H
#define HPSC_MPI_EIGEN_HELPERS_H

#include <memory>

#include "common/eigen_helpers.h"
#include "common/mpi_helpers.h"

/**
 * @brief Broadcasts the given sparse Eigen matrix using MPI, starting from the given node.
 * @param A         The sparse matrix to broadcast.
 * @param sender    The index of the root node.
 */
void BroadcastEigenSparse(ESMatrix &A, int sender = 0);

/**
 *  @brief Transposes a square row-wise distributed matrix using MPI.
 *  @details Each of the p nodes holds a [n/p x m] chunk of a matrix.
 *  @param[in] in_chunk     [n/p x m] matrix owned locally by a processor.
 *  @param[out] out         [m/p x n] matrix corresponding to the transposed result.
 */
void TransposeEigenDense(const EMatrix &in_chunk, EMatrix &out);


#endif //HPSC_MPI_EIGEN_HELPERS_H
