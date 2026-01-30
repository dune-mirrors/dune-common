# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

if(DUNE_USE_SYSTEM_PYBIND11)
  # handle the extra file extensions.h
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/extensions.h
    ${PROJECT_BINARY_DIR}/dune/python/pybind11/extensions.h COPYONLY)
  install(FILES ${PROJECT_BINARY_DIR}/dune/python/pybind11/extensions.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/dune/python/pybind11)
  target_include_directories(dunecommon INTERFACE
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external>)

  # add the old directory name as fallback, using symbolic links
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/dune/python)
  file(CREATE_LINK ${pybind11_INCLUDE_DIR}/pybind11
    ${PROJECT_BINARY_DIR}/include/dune/python/pybind11 SYMBOLIC)
  install(FILES ${PROJECT_BINARY_DIR}/include/dune/python/pybind11
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/python)
else()
  add_subdirectory(pybind11)
endif()