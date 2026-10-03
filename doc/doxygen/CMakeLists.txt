# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# create Doxyfile.in and Doxyfile
add_doxygen_target()

install(
  FILES
    Doxystyle
    doxygen-macros
  DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/dune/doxygen
  )

# Doxygen themes provided by dune-common
dune_add_doxygen_theme(NAME awesome)
install(
  FILES
    ${PROJECT_SOURCE_DIR}/LICENSES/MIT.txt
  DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/dune/doxygen/themes/awesome
  RENAME
    LICENSE
  )
