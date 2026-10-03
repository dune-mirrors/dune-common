# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#[=======================================================================[.rst:
DuneDoxygen
===========

Support for building module documentation with Doxygen.

.. cmake:command:: add_doxygen_target

  Create a Doxygen build target for the current module.

  .. code-block:: cmake

    add_doxygen_target(
      [TARGET <target-suffix>]
      [DEPENDS <files>...]
      [OUTPUT <path>]
      [THEME <name>]
    )

  ``TARGET``
    Suffix of the generated build target. The default is the current module
    name.

  ``DEPENDS``
    Additional dependencies of the generated Doxygen build step, for example a
    manually maintained ``mainpage.txt`` file.

  ``OUTPUT``
    Output path produced by the Doxygen run. The default is the generated
    ``html`` directory in the current binary directory.

  ``THEME``
    Name of a Doxygen theme, e.g., ``awesome``. The theme is searched in the
    current module and in all its dependencies, see
    :cmake:command:`dune_add_doxygen_theme`. The default is the classic
    Doxygen look. The cache variable :cmake:variable:`DUNE_DOXYGEN_THEME`
    overrides this choice.

  This command creates a module-specific ``doxygen_<target>`` target and adds
  it as a dependency of the top-level ``doxygen`` and ``doc`` targets. During
  installation, the generated Doxygen output is copied into
  ``${CMAKE_INSTALL_DOCDIR}/doxygen``.

.. cmake:command:: dune_add_doxygen_theme

  Install a Doxygen theme provided by the current module, such that modules
  depending on it can select the theme with the ``THEME`` argument of
  :cmake:command:`add_doxygen_target`.

  .. code-block:: cmake

    dune_add_doxygen_theme(NAME <name>)

  ``NAME``
    Name of the theme.

  The files of the theme are located in ``doc/doxygen/themes/<name>/`` of
  the module's source directory and are installed to
  ``${CMAKE_INSTALL_DATAROOTDIR}/dune/doxygen/themes/<name>/``. The theme
  directory has to contain a Doxyfile fragment ``Doxyfile.theme`` with the
  settings of the theme. The fragment is inserted between the global
  Doxystyle and the module's ``Doxylocal``, such that modules can still
  override single settings. Within the fragment,
  ``@DUNE_DOXYGEN_THEME_DIR@`` refers to the theme directory.

  :cmake:command:`add_doxygen_target` searches a theme at these canonical
  locations: in the source directory of the current module, and for each
  dependency either in its source directory (if it is used from its build
  tree) or below its installation prefix. Installed modules are assumed to
  use the same ``CMAKE_INSTALL_DATAROOTDIR``. It is an error if a theme is
  provided by more than one module.

  dune-common provides the theme ``awesome``, based on
  `doxygen-awesome-css <https://jothepro.github.io/doxygen-awesome-css>`_.

.. cmake:variable:: DUNE_DOXYGEN_THEME

  If set, this Doxygen theme is used for all modules, overriding the
  ``THEME`` argument of :cmake:command:`add_doxygen_target`. The value
  ``none`` enforces the classic Doxygen look, which is otherwise used for
  all modules that do not specify a theme.

.. cmake:variable:: DUNE_MATHJAX_DISABLE_LOCAL

  If set to ``TRUE``, local MathJax discovery is disabled and Doxygen will not
  use an installed MathJax2 copy from the system.

.. cmake:variable:: DUNE_MATHJAX_DISABLE_CDN

  If set to ``TRUE``, MathJax will not be loaded from the content delivery
  network when no local MathJax2 installation is available.

#]=======================================================================]
include_guard(GLOBAL)

find_package(Doxygen)
set_package_properties("Doxygen" PROPERTIES
  DESCRIPTION "Class documentation generator"
  URL "www.doxygen.org"
  PURPOSE "To generate the class documentation from C++ sources")

# Set DOT_TRUE for the Doxyfile generation.
if (NOT DOXYGEN_DOT_FOUND)
  set(DOT_TRUE '\#')
endif()

add_custom_target(doxygen)
add_dependencies(doc doxygen)
add_custom_target(doxygen_install)

##############################
# Begin MathJax support

# Variables to configure MathJax support
option(DUNE_MATHJAX_DISABLE_LOCAL "Flag to disable usage of local MathJax")
option(DUNE_MATHJAX_DISABLE_CDN "Flag to disable usage of MathJax from content delivery network")

# Use local MathJax2 unless disabled
set(mathjax_relpath "")
set(use_mathjax OFF)
if(NOT DUNE_MATHJAX_DISABLE_LOCAL)
  # This currently searches for MathJax2 only which is the default in Doxygen.
  # Newer versions do not provide MathJax.js and have to be enabled manually
  # in Doxygen.
  find_package(MathJax2)
  if(MATHJAX2_FOUND)
    message(STATUS "Using local MathJax found in ${MATHJAX2_PATH}")
    set(use_mathjax ON)
    set(mathjax_relpath "${MATHJAX2_PATH}")
  endif()
endif()

# Use MathJax2 from cdn unless disabled
if((NOT DUNE_MATHJAX_DISABLE_CDN) AND (mathjax_relpath STREQUAL ""))
  message(STATUS "Using MathJax from content delivery network")
  set(use_mathjax ON)
endif()

# Don't use MathJax
if(NOT use_mathjax)
  message(STATUS "MathJax is disabled")
endif()

# Variables forwarded to Doxygen
set_property(GLOBAL PROPERTY DUNE_USE_MATHJAX "${use_mathjax}")
set_property(GLOBAL PROPERTY DUNE_MATHJAX_RELPATH "${mathjax_relpath}")

# End MathJax support
##############################

##############################
# Begin Doxygen themes

set(DUNE_DOXYGEN_THEME "" CACHE STRING
  "Doxygen theme used for all modules, overriding the module's choice ('none' for the classic look)")

function(dune_add_doxygen_theme)
  cmake_parse_arguments(THEME "" "NAME" "" ${ARGN})
  if(THEME_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments in dune_add_doxygen_theme: ${THEME_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT THEME_NAME)
    message(FATAL_ERROR "dune_add_doxygen_theme requires NAME")
  endif()
  set(_directory "${PROJECT_SOURCE_DIR}/doc/doxygen/themes/${THEME_NAME}")
  if(NOT EXISTS "${_directory}/Doxyfile.theme")
    message(FATAL_ERROR "Doxygen theme '${THEME_NAME}' not found: ${_directory}/Doxyfile.theme does not exist")
  endif()
  install(DIRECTORY "${_directory}/"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dune/doxygen/themes/${THEME_NAME})
endfunction()

# Find the directory of the Doxygen theme <name> in the current module or
# one of its dependencies and store it in <result>.
function(dune_find_doxygen_theme name result)
  set(_candidates "${PROJECT_NAME}|${PROJECT_SOURCE_DIR}/doc/doxygen/themes")
  foreach(_mod IN LISTS DUNE_FOUND_DEPENDENCIES)
    if(${_mod}_INSTALLED)
      list(APPEND _candidates "${_mod}|${${_mod}_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/dune/doxygen/themes")
    else()
      list(APPEND _candidates "${_mod}|${${_mod}_PREFIX}/doc/doxygen/themes")
    endif()
  endforeach()

  set(_searched "")
  set(_found_dirs "")
  set(_found "")
  foreach(_candidate IN LISTS _candidates)
    string(REPLACE "|" ";" _candidate "${_candidate}")
    list(GET _candidate 0 _mod)
    list(GET _candidate 1 _themes_dir)
    list(APPEND _searched "${_themes_dir}")
    if(EXISTS "${_themes_dir}/${name}/Doxyfile.theme")
      get_filename_component(_dir "${_themes_dir}/${name}" REALPATH)
      if(NOT _dir IN_LIST _found_dirs)
        list(APPEND _found_dirs "${_dir}")
        list(APPEND _found "  ${_dir} (found via ${_mod})")
      endif()
    endif()
  endforeach()

  list(LENGTH _found_dirs _count)
  if(_count EQUAL 0)
    list(REMOVE_DUPLICATES _searched)
    list(JOIN _searched "\n  " _searched)
    message(FATAL_ERROR "Unknown Doxygen theme '${name}', searched in:\n  ${_searched}")
  elseif(_count GREATER 1)
    list(JOIN _found "\n" _found)
    message(FATAL_ERROR "Doxygen theme '${name}' is provided more than once:\n${_found}")
  endif()
  set(${result} "${_found_dirs}" PARENT_SCOPE)
endfunction()

# End Doxygen themes
##############################

#
# prepare_doxyfile()
# This functions adds the necessary routines for the generation of the
# Doxyfile[.in] files needed to doxygen.
macro(prepare_doxyfile)
  cmake_parse_arguments(DOXYFILE "" "TARGET;DOXYFILE;THEME_DIR" "TAGFILES" ${ARGN})

  # default target name is the module name
  if(NOT DOXYFILE_TARGET)
    set(DOXYFILE_TARGET ${PROJECT_NAME})
  endif()

  # Get global properties for MathJax configuration
  get_property(use_mathjax GLOBAL PROPERTY DUNE_USE_MATHJAX)
  get_property(mathjax_relpath GLOBAL PROPERTY DUNE_MATHJAX_RELPATH)

  set(DUNE_DOXYGEN_TOPICSLINK "[Topics](topics.html)")
  if(DOXYGEN_VERSION VERSION_LESS 1.9.8)
    set(DUNE_DOXYGEN_TOPICSLINK "[Modules](modules.html)")
  endif()

  message(STATUS "using ${DOXYSTYLE_FILE} to create doxystyle file")
  message(STATUS "using C macro definitions from ${DOXYGENMACROS_FILE} for Doxygen")

  # check whether module has a Doxylocal file
  find_file(_DOXYLOCAL Doxylocal PATHS ${CMAKE_CURRENT_SOURCE_DIR} NO_DEFAULT_PATH)
  set(make_doxyfile_options
    -D DOT_TRUE=${DOT_TRUE}
    -D DUNE_MOD_NAME=${PROJECT_NAME}
    -D DUNE_MOD_VERSION=${ProjectVersion}
    -D DOXYSTYLE=${DOXYSTYLE_FILE}
    -D DOXYGENMACROS=${DOXYGENMACROS_FILE}
    -D abs_top_srcdir=${CMAKE_SOURCE_DIR}
    -D top_srcdir=${${PROJECT_NAME}_SOURCE_DIR}
    -D DOXYGEN_TAGFILES=${DOXYFILE_TAGFILES}
    -D DUNE_USE_MATHJAX=$<IF:$<BOOL:${use_mathjax}>,YES,NO>
    -D DUNE_DOXYGEN_TOPICSLINK="${DUNE_DOXYGEN_TOPICSLINK}"
    -D DUNE_MATHJAX_RELPATH="${mathjax_relpath}"
    -D DOXYFILE=${DOXYFILE_DOXYFILE})
  set(_doxyfile_depends ${DOXYSTYLE_FILE} ${DOXYGENMACROS_FILE})
  if(DOXYFILE_THEME_DIR)
    list(APPEND make_doxyfile_options
      -D DOXYTHEME=${DOXYFILE_THEME_DIR}/Doxyfile.theme
      -D DUNE_DOXYGEN_THEME_DIR=${DOXYFILE_THEME_DIR})
    list(APPEND _doxyfile_depends ${DOXYFILE_THEME_DIR}/Doxyfile.theme)
  endif()
  list(APPEND make_doxyfile_options -P ${scriptdir}/CreateDoxyFile.cmake)
  if(_DOXYLOCAL)
    add_custom_command(OUTPUT ${DOXYFILE_DOXYFILE}.in ${DOXYFILE_DOXYFILE}
      COMMAND ${CMAKE_COMMAND} ${make_doxyfile_options} -D DOXYLOCAL=${CMAKE_CURRENT_SOURCE_DIR}/Doxylocal -D srcdir=${CMAKE_CURRENT_SOURCE_DIR} -P ${scriptdir}/CreateDoxyFile.cmake
      COMMENT "Creating ${DOXYFILE_DOXYFILE}.in"
      DEPENDS ${_doxyfile_depends} ${CMAKE_CURRENT_SOURCE_DIR}/Doxylocal)
  else()
    add_custom_command(OUTPUT ${DOXYFILE_DOXYFILE}.in ${DOXYFILE_DOXYFILE}
      COMMAND ${CMAKE_COMMAND}  ${make_doxyfile_options} -P ${scriptdir}/CreateDoxyFile.cmake
      COMMENT "Creating ${DOXYFILE_DOXYFILE}.in"
      DEPENDS ${_doxyfile_depends})
  endif()
  add_custom_target(doxyfile_${DOXYFILE_TARGET} DEPENDS ${DOXYFILE_DOXYFILE}.in ${DOXYFILE_DOXYFILE})
endmacro(prepare_doxyfile)

macro(add_doxygen_target)
  set(options )
  set(oneValueArgs TARGET OUTPUT THEME)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(DOXYGEN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  # default target name is the module name
  if(NOT DOXYGEN_TARGET)
    set(DOXYGEN_TARGET ${PROJECT_NAME})
  endif()

  # the theme selected by the user overrides the one of the module
  if(DUNE_DOXYGEN_THEME)
    set(DOXYGEN_THEME ${DUNE_DOXYGEN_THEME})
  endif()
  if(DOXYGEN_THEME STREQUAL "none")
    unset(DOXYGEN_THEME)
  endif()
  set(_doxygen_theme_dir "")
  if(DOXYGEN_THEME)
    dune_find_doxygen_theme(${DOXYGEN_THEME} _doxygen_theme_dir)
    message(STATUS "Using Doxygen theme '${DOXYGEN_THEME}' from ${_doxygen_theme_dir}")
    # rerun Doxygen if a file of the theme changes
    file(GLOB _theme_files "${_doxygen_theme_dir}/*")
    list(APPEND DOXYGEN_DEPENDS ${_theme_files})
  endif()

  dune_module_path(MODULE dune-common RESULT scriptdir SCRIPT_DIR)
  if(PROJECT_NAME STREQUAL "dune-common")
    set(DOXYSTYLE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Doxystyle)
    set(DOXYGENMACROS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-macros)
  endif()
  message(STATUS "Using scripts from ${scriptdir} for creating doxygen stuff.")

  if(TARGET Doxygen::doxygen)

    set(DOXYGEN_BUILD_TAGFILES ${DOXYGEN_TAGFILES})
    foreach(module ${DUNE_FOUND_DEPENDENCIES})
      set(DOXYGEN_BUILD_TAGFILES "${DOXYGEN_BUILD_TAGFILES} ${${module}_DOXYGEN_DIR}/${module}.tag=${${module}_DOXYGEN_DIR}/html")
    endforeach()
    prepare_doxyfile(TARGET ${DOXYGEN_TARGET} TAGFILES ${DOXYGEN_BUILD_TAGFILES} DOXYFILE Doxyfile THEME_DIR "${_doxygen_theme_dir}")
    # custom command that executes doxygen
    add_custom_command(OUTPUT ${DOXYGEN_OUTPUT} html/ ${PROJECT_NAME}.tag
      COMMAND ${CMAKE_COMMAND} -D DOXYGEN_EXECUTABLE=$<TARGET_FILE:Doxygen::doxygen> -D DOXYFILE=Doxyfile -P ${scriptdir}/RunDoxygen.cmake
      COMMENT "Building doxygen documentation. This may take a while"
      DEPENDS Doxyfile ${DOXYGEN_DEPENDS})
    # Create a target for building the doxygen documentation of a module,
    # that is run during make doc
    add_custom_target(doxygen_${DOXYGEN_TARGET} DEPENDS html ${PROJECT_NAME}.tag)
    add_dependencies(doxygen doxygen_${DOXYGEN_TARGET})
    foreach(module ${DUNE_FOUND_DEPENDENCIES})
      get_property(module_doxygen_target GLOBAL PROPERTY ${module}_DOXYGEN_TARGET)
      if(TARGET ${module_doxygen_target})
        add_dependencies(doxygen_${DOXYGEN_TARGET} ${module_doxygen_target})
      endif()
    endforeach()

    set_property(GLOBAL PROPERTY ${PROJECT_NAME}_DOXYGEN_TARGET doxygen_${DOXYGEN_TARGET})
    set_property(GLOBAL PROPERTY ${PROJECT_NAME}_DOXYGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    # Use a cmake call to install the doxygen documentation and create a
    # target for it
    include(GNUInstallDirs)

    set(DOXYGEN_INSTALL_TAGFILES ${DOXYGEN_TAGFILES})
    foreach(module ${DUNE_FOUND_DEPENDENCIES})
      if(${module}_INSTALLED)
        # The module is already installed, so we can directly use the doxygen directories from the config file
        set(DOXYGEN_INSTALL_TAGFILES "${DOXYGEN_INSTALL_TAGFILES} ${${module}_DOXYGEN_DIR}/${module}.tag=${${module}_DOXYGEN_DIR}/html")
      else()
        # The module is not yet installed, so we assume is going to be installed on the same prefix as this module.
        set(DOXYGEN_INSTALL_TAGFILES "${DOXYGEN_INSTALL_TAGFILES} ${${module}_DOXYGEN_DIR}/installdir/${module}.tag=../../../${module}/doxygen/html")
      endif()
    endforeach()

    prepare_doxyfile(TARGET ${DOXYGEN_TARGET}_install TAGFILES ${DOXYGEN_INSTALL_TAGFILES} DOXYFILE installdir/Doxyfile THEME_DIR "${_doxygen_theme_dir}")
    # custom command that executes doxygen
    add_custom_command(OUTPUT ${DOXYGEN_OUTPUT} installdir/html/ installdir/${PROJECT_NAME}.tag
      COMMAND ${CMAKE_COMMAND} -D DOXYGEN_EXECUTABLE=$<TARGET_FILE:Doxygen::doxygen> -D DOXYFILE=Doxyfile -P ${scriptdir}/RunDoxygen.cmake
      COMMENT "Building doxygen documentation. This may take a while"
      WORKING_DIRECTORY installdir
      DEPENDS installdir/Doxyfile ${DOXYGEN_DEPENDS})

    add_custom_target(doxygen_${DOXYGEN_TARGET}_install
      DEPENDS installdir/html/ installdir/${PROJECT_NAME}.tag)

    foreach(module ${DUNE_FOUND_DEPENDENCIES})
      get_property(module_doxygen_target GLOBAL PROPERTY ${module}_DOXYGEN_TARGET)
      if(TARGET ${module_doxygen_target})
        add_dependencies(doxygen_${DOXYGEN_TARGET}_install ${module_doxygen_target}_install)
      endif()
    endforeach()

    # When installing call cmake install with the above install target
    install(CODE
      "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target doxygen_${ProjectName}_install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/installdir/html/" DESTINATION ${CMAKE_INSTALL_DOCDIR}/doxygen/html)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installdir/${PROJECT_NAME}.tag" DESTINATION ${CMAKE_INSTALL_DOCDIR}/doxygen)
  endif()
endmacro(add_doxygen_target)
