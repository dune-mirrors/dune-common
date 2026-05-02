# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#[=======================================================================[.rst:
DuneModuleLibrary
-----------------

Initialize a Dune module and create its module library target.

.. cmake:command:: dune_module_library

  Initialize the current DUNE module from its top-level ``CMakeLists.txt`` and
  create the library target that represents the module.

  .. code-block:: cmake

    dune_module_library(<basename> [STATIC|SHARED|MODULE]
      [SOURCES <sources...>]
      [LINK_LIBRARIES <targets>...]
      [COMPILE_OPTIONS "<flags>;..."]
      [OUTPUT_NAME <libname>]
      EXPORT_NAME <exportname>
      [NAMESPACE <namespace>]
      [LINK_SUGGESTED_LIBRARIES]
    )

  This command is intended as a replacement for a top-level combination of
  :cmake:command:`dune_project()` and :cmake:command:`dune_add_library()`. It
  reads the local ``dune.module`` file, initializes the standard ``Project*``
  variables, finds required and suggested DUNE dependencies, enables the
  documentation infrastructure, and creates the module library target.

  Unlike :cmake:command:`dune_project()`, this command does not process
  dependency macro files and relies on the current module's ``dune.module``
  metadata together with dependency package configuration files.

  The created target is exported like a regular dune library and receives the
  compile definition ``HAVE_<MODULE_NAME>=1`` where ``<MODULE_NAME>`` is the
  uppercase module name with dashes replaced by underscores.
  Libraries exported by modules listed in ``Depends:`` are linked
  automatically. Found modules listed in ``Suggests:`` are linked only if
  ``LINK_SUGGESTED_LIBRARIES`` is passed.

  ``SOURCES``
    The source files from which to build the library.

  ``LINK_LIBRARIES`` (or ``ADD_LIBS``)
    A list of libraries the target is explicitly linked against.

  ``COMPILE_OPTIONS`` (or ``COMPILE_FLAGS``)
    Additional compile flags for building the library.

  ``OUTPUT_NAME``
    Name of the created library file, e.g. ``lib<libname>.so`` or
    ``lib<libname>.a``.

  ``NAMESPACE``
    Name to be prepended to the export name of the target.
    By default this is set to ``Dune::``.

  ``EXPORT_NAME``
    Name of the exported target to be used when linking against the library.
    The name is prepended with the given namespace.

  ``LINK_SUGGESTED_LIBRARIES``
    Link module libraries exported by found modules listed in ``Suggests:``.

  .. code-block:: cmake

    dune_module_library(<basename> INTERFACE
      [LINK_LIBRARIES <targets>...]
      [COMPILE_OPTIONS "<flags>;..."]
      EXPORT_NAME <exportname>
      [NAMESPACE <namespace>]
      [LINK_SUGGESTED_LIBRARIES]
    )

  Create an interface module library target. Interface module libraries are
  exported like regular module libraries and receive the same
  ``HAVE_<MODULE_NAME>=1`` definition.

  Call :cmake:command:`finalize_dune_project()` at the end of the same
  top-level ``CMakeLists.txt`` file.

#]=======================================================================]
include_guard(GLOBAL)

include(CMakeDependentOption)
include(CheckCXXFeatures)
include(DuneAddLibrary)
include(DuneDefaultIncludeDirectories)
include(DuneDoc)
include(DuneModuleDependencies)
include(DuneModuleInformation)
include(GNUInstallDirs)

if(NOT DEFINED DUNE_ENABLE_TESTING)
  option(DUNE_ENABLE_TESTING "Enable test subdirectories in dune modules by default." ON)
endif()


macro(_dune_initialize_module_library_project)
  if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(DEPRECATION "The function `dune_module_library` shall only be called on CMake project root directory.
    Note that CMake projects may be nested: https://cmake.org/cmake/help/latest/command/project.html.")
  endif()

  dune_module_information(${PROJECT_SOURCE_DIR})
  set(ProjectName            "${DUNE_MOD_NAME}")
  set(ProjectVersion         "${DUNE_MOD_VERSION}")
  set(ProjectVersionString   "${DUNE_VERSION_MAJOR}.${DUNE_VERSION_MINOR}.${DUNE_VERSION_REVISION}")
  set(ProjectVersionMajor    "${DUNE_VERSION_MAJOR}")
  set(ProjectVersionMinor    "${DUNE_VERSION_MINOR}")
  set(ProjectVersionRevision "${DUNE_VERSION_REVISION}")
  set(ProjectMaintainerEmail "${DUNE_MAINTAINER}")
  set(ProjectDescription     "${${DUNE_MOD_NAME_UPPERCASE}_DESCRIPTION}")
  set(ProjectAuthor          "${${DUNE_MOD_NAME_UPPERCASE}_AUTHOR}")
  set(ProjectUrl             "${${DUNE_MOD_NAME_UPPERCASE}_URL}")
  set(ProjectPythonRequires  "${${DUNE_MOD_NAME_UPPERCASE}_PYTHON_REQUIRES}")

  set(_module_testing_option "${DUNE_MOD_NAME_UPPERCASE}_ENABLE_TESTING")
  option(${_module_testing_option}
    "Enable test subdirectories for the ${ProjectName} module."
    ${DUNE_ENABLE_TESTING})
  set(DUNE_MODULE_ENABLE_TESTING ${${_module_testing_option}})
  unset(_module_testing_option)

  if(CMAKE_DISABLE_FIND_PACKAGE_${ProjectName})
    message("Module ${ProjectName} has been explicitly disabled through the cmake flags. Skipping build.")
    return()
  endif()

  define_property(GLOBAL PROPERTY ${ProjectName}_LIBRARIES
    BRIEF_DOCS "List of libraries of the module. DO NOT EDIT!"
    FULL_DOCS "List of all module-libraries. DO NOT EDIT!")

  define_property(GLOBAL PROPERTY ${ProjectName}_EXPORTED_LIBRARIES
    BRIEF_DOCS "List of libraries exported by the module. DO NOT EDIT!"
    FULL_DOCS "List of libraries exported by the module. DO NOT EDIT!")

  foreach(_mod IN LISTS ${ProjectName}_SUGGESTS)
    split_module_version(${_mod} _mod_name _mod_ver)
    find_dune_package(${_mod_name} VERSION ${_mod_ver} QUIET)
  endforeach()

  foreach(_mod IN LISTS ${ProjectName}_DEPENDS)
    split_module_version(${_mod} _mod_name _mod_ver)
    find_dune_package(${_mod_name} VERSION ${_mod_ver} REQUIRED)
  endforeach()

  if(NOT (ProjectName STREQUAL PROJECT_NAME))
    message(FATAL_ERROR "Module name from dune.module does not match the name given in CMakeLists.txt.")
  endif()

  cmake_dependent_option(CMAKE_POSITION_INDEPENDENT_CODE "Build position independent code" ON "NOT BUILD_SHARED_LIBS" ON)

  dune_cmake_path_setup(DUNE_FOUND_DEPENDENCIES)

  if(NOT DUNE_INSTALL_MODULEDIR)
    set(DUNE_INSTALL_MODULEDIR ""
      CACHE PATH
      "Installation directory for CMake modules. Default is \${CMAKE_INSTALL_DATAROOTDIR}/dune/cmake/modules when not set explicitly")
    set(DUNE_INSTALL_MODULEDIR ${CMAKE_INSTALL_DATAROOTDIR}/dune/cmake/modules)
  endif()
  if(NOT DUNE_INSTALL_NONOBJECTLIBDIR)
    set(DUNE_INSTALL_NONOBJECTLIBDIR ""
      CACHE PATH
      "Installation directory for libraries that are not architecture dependent. Default is lib when not set explicitly")
    set(DUNE_INSTALL_NONOBJECTLIBDIR lib)
  endif()

  set(${ProjectName}_FOUND 1)
endmacro()


function(_dune_module_library_dependency_libraries _result_var _link_suggested_libraries)
  set(_dune_module_library_libraries "")

  foreach(_mod IN LISTS ${ProjectName}_DEPENDS)
    split_module_version(${_mod} _mod_name _mod_ver)
    if(${_mod_name}_FOUND AND ${_mod_name}_LIBRARIES)
      list(APPEND _dune_module_library_libraries ${${_mod_name}_LIBRARIES})
    endif()
  endforeach()

  if(_link_suggested_libraries)
    foreach(_mod IN LISTS ${ProjectName}_SUGGESTS)
      split_module_version(${_mod} _mod_name _mod_ver)
      if(${_mod_name}_FOUND AND ${_mod_name}_LIBRARIES)
        list(APPEND _dune_module_library_libraries ${${_mod_name}_LIBRARIES})
      endif()
    endforeach()
  endif()

  list(REMOVE_DUPLICATES _dune_module_library_libraries)
  set(${_result_var} "${_dune_module_library_libraries}" PARENT_SCOPE)
endfunction()


macro(dune_module_library _name)
  list(FIND ARGN "OBJECT" _dune_module_library_object_index)
  if(NOT _dune_module_library_object_index EQUAL -1)
    message(FATAL_ERROR "dune_module_library does not support OBJECT libraries.")
  endif()

  cmake_parse_arguments(ARG
    "NO_EXPORT;NO_MODULE_LIBRARY;LINK_SUGGESTED_LIBRARIES;INTERFACE;STATIC;SHARED;MODULE"
    "COMPILE_FLAGS;COMPILE_OPTIONS;OUTPUT_NAME;EXPORT_NAME;NAMESPACE"
    "ADD_LIBS;LINK_LIBRARIES;SOURCES" ${ARGN})

  if(ARG_NO_EXPORT)
    message(FATAL_ERROR "dune_module_library always exports the module library. Remove NO_EXPORT.")
  endif()

  if(ARG_NO_MODULE_LIBRARY)
    message(FATAL_ERROR "dune_module_library always registers the module library. Remove NO_MODULE_LIBRARY.")
  endif()

  if(NOT ARG_EXPORT_NAME)
    message(FATAL_ERROR "You must provide EXPORT_NAME to dune_module_library.")
  endif()

  if(NOT DEFINED ProjectName OR NOT ProjectName STREQUAL PROJECT_NAME)
    _dune_initialize_module_library_project()
  endif()

  set(_dune_module_library_args ${ARGN})
  if(NOT ARG_NAMESPACE)
    list(APPEND _dune_module_library_args NAMESPACE Dune::)
  endif()

  dune_add_library(${_name} ${_dune_module_library_args})

  _dune_module_library_dependency_libraries(_dune_module_library_deps ${ARG_LINK_SUGGESTED_LIBRARIES})
  if(_dune_module_library_deps)
    if(ARG_INTERFACE)
      target_link_libraries(${_name} INTERFACE ${_dune_module_library_deps})
    else()
      target_link_libraries(${_name} PUBLIC ${_dune_module_library_deps})
    endif()
  endif()

  dune_module_to_uppercase(_dune_module_upper "${ProjectName}")
  if(ARG_INTERFACE)
    dune_default_include_directories(${_name} INTERFACE)
    target_compile_definitions(${_name} INTERFACE
      HAVE_CONFIG_H=1
      HAVE_${_dune_module_upper}=1)
  else()
    dune_default_include_directories(${_name} PUBLIC)
    target_compile_definitions(${_name} PUBLIC
      HAVE_CONFIG_H=1
      HAVE_${_dune_module_upper}=1)
  endif()
endmacro(dune_module_library)
