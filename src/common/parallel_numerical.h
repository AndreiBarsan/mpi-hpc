//
// Implements parallel solvers.
//

#ifndef HPSC_PARALLEL_NUMERICAL_H
#define HPSC_PARALLEL_NUMERICAL_H

#include "serial_numerical.h"
#include "mpi_helpers.h"

#ifdef DEBUG_WITH_EIGEN
/// Debug serial implementation of parallel partitioning method II for tridiagonal systems..
/// \see SolveParallel for more information.
Matrix<double> SolveParallelDebug(const BandMatrix<double> &A_raw, const Matrix<double> &b_raw);
#endif    // DEBUG_WITH_EIGEN

/// \brief Solves the given tridiagonal system AX = B using MPI a block-based partitioning method.
///
/// \details The algorithm should also work on arbitrary banded matrices with only small modifications in the way the
/// data transfers are performed in the beginning. X and B can have multiple columns, i.e., multiple right-hand sides
/// are supported.
///
/// The partitioning scheme used is known as Johnsson's partitioning method, and is the second one detailed in Ortega
/// '88, Section 2.3, subsection "Partitioning Methods"  (p. 118, Figure 2.3.12). The book explains it quite well; I
/// strongly suggest checking it out!
///
/// \param A An n x n banded matrix.
/// \param b An n x m matrix (m right-hand side vectors).
/// \param b_is_distributed Whether to expect the b-matrix to already be split up row-wise.
/// \param verbose Whether to print out various bits of info while executing.
/// \return X, the n x m solution vector.
Matrix<double> SolveParallel(const BandMatrix<double> &A,
                             const Matrix<double> &b,
                             bool b_is_distributed = false,
                             bool verbose = false);

#endif //HPSC_PARALLEL_NUMERICAL_H
