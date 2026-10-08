# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Deliberately standalone: the control program must not link against DUNE.
cmake_minimum_required(VERSION 3.16)
project(dune_mpi_finalize_diagnostics LANGUAGES C)
find_package(MPI REQUIRED COMPONENTS C)

add_library(mpi_finalize_trace SHARED trace.c)
target_link_libraries(mpi_finalize_trace PRIVATE MPI::MPI_C)
target_compile_features(mpi_finalize_trace PRIVATE c_std_99)

add_executable(mpi_finalize_probe probe.c)
target_link_libraries(mpi_finalize_probe PRIVATE MPI::MPI_C)
target_compile_features(mpi_finalize_probe PRIVATE c_std_99)
