# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

add_subdirectory(archetypes)

install(FILES
  container.hh
  hashable.hh
  multidimensional.hh
  number.hh
DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/common/concepts)
