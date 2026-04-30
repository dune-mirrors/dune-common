# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

SET(HEADERS
class.h  common.h  descr.h  init.h  internals.h  type_caster_base.h  typeid.h
)

if(NOT DUNE_USE_SYSTEM_PYBIND11)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/python/pybind11/detail)
endif()
