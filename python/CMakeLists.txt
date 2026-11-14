# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

add_subdirectory(dune)
add_subdirectory(
  template
  "${CMAKE_CURRENT_BINARY_DIR}/dune/generator/template"
)


dune_python_configure_bindings(
  PATH "."
  CMAKE_METADATA_FLAGS DUNE_OPTS_FILE HAVE_MPI CMAKE_BUILD_TYPE CMAKE_CXX_COMPILER CMAKE_CXX_FLAGS CMAKE_C_COMPILER CMAKE_C_FLAGS CMAKE_PREFIX_PATH CMAKE_CXX_COMPILER_LAUNCHER DUNE_USE_SYSTEM_PYBIND11
)
