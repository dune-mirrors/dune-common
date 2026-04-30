# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

SET(HEADERS
common.h  matrix.h  tensor.h
)

if(NOT DUNE_USE_SYSTEM_PYBIND11)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/python/pybind11/eigen)
endif()
