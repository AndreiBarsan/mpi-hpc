#ifndef HPSC_MPI_HELPERS_H
#define HPSC_MPI_HELPERS_H

#include "mpi.h"

#define MPI_SETUP  \
  int local_id, n_procs; \
  MPI_Comm_rank(MPI_COMM_WORLD, &local_id); \
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#define MASTER if (0 == local_id)



#endif //HPSC_MPI_HELPERS_H
