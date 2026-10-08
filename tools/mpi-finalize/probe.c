// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include <mpi.h>
#include <stdio.h>

/* No DUNE, C++ static objects, or application communication. */
int main(int argc, char **argv)
{
  int rank = -1;
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS)
    return 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  fprintf(stderr, "probe rank=%d entering MPI_Finalize\n", rank);
  fflush(stderr);
  rc = MPI_Finalize();
  fprintf(stderr, "probe rank=%d returned from MPI_Finalize rc=%d\n", rank, rc);
  fflush(stderr);
  return rc == MPI_SUCCESS ? 0 : 1;
}
