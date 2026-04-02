# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Defines the functions to use GMP
#
# .. cmake_function:: add_dune_gmp_flags
#
#    .. cmake_param:: targets
#       :positional:
#       :single:
#       :required:
#
#       A list of targets to use GMP with.
#
include_guard(GLOBAL)

function(_dune_gmp_select_backend _outvar)
  set(options)
  set(oneValueArgs BACKEND)
  set(multiValueArgs)
  cmake_parse_arguments(ADD_GMP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ADD_GMP_BACKEND)
    set(ADD_GMP_BACKEND AUTO)
  endif()

  if(ADD_GMP_BACKEND STREQUAL "AUTO")
    if(MPFR_FOUND)
      set(_backend MPFR)
    elseif(GMP_FOUND)
      set(_backend GMPXX)
    else()
      set(_backend NONE)
    endif()
  elseif(ADD_GMP_BACKEND STREQUAL "MPFR")
    if(NOT MPFR_FOUND)
      message(FATAL_ERROR "Requested GMP backend MPFR, but MPFR was not found.")
    endif()
    set(_backend MPFR)
  elseif(ADD_GMP_BACKEND STREQUAL "GMPXX")
    if(NOT GMP_FOUND)
      message(FATAL_ERROR "Requested GMP backend GMPXX, but GMPXX was not found.")
    endif()
    set(_backend GMPXX)
  else()
    message(FATAL_ERROR "Unknown BACKEND='${ADD_GMP_BACKEND}' in add_dune_gmp_flags. Supported values are AUTO, MPFR, and GMPXX.")
  endif()

  set(${_outvar} ${_backend} PARENT_SCOPE)
endfunction()

# set HAVE_GMP for the config.h file
if(GMP_FOUND OR MPFR_FOUND)
  set(HAVE_GMP TRUE)
else()
  set(HAVE_GMP FALSE)
endif()

# register all GMP related flags
_dune_gmp_select_backend(_dune_default_gmp_backend BACKEND AUTO)
if(_dune_default_gmp_backend STREQUAL "MPFR")
  dune_register_package_flags(
    LIBRARIES MPFR::mpreal
    COMPILE_DEFINITIONS "HAVE_GMP=1;HAVE_MPFR=1"
  )
elseif(_dune_default_gmp_backend STREQUAL "GMPXX")
  message(WARNING "The library GMPXX is outdated and deprecated. Use MPFR C++ (mpreal)  instead.")
  dune_register_package_flags(
    LIBRARIES GMP::gmpxx
    COMPILE_DEFINITIONS "HAVE_GMP=1;HAVE_GMPXX=1"
  )
endif()

# add function to link against the GMP library
function(add_dune_gmp_flags)
  set(options)
  set(oneValueArgs BACKEND)
  set(multiValueArgs TARGETS)
  cmake_parse_arguments(ADD_GMP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(ADD_GMP_UNPARSED_ARGUMENTS)
    list(APPEND ADD_GMP_TARGETS ${ADD_GMP_UNPARSED_ARGUMENTS})
  endif()

  if(NOT ADD_GMP_TARGETS)
    message(FATAL_ERROR "add_dune_gmp_flags requires at least one target.")
  endif()

  _dune_gmp_select_backend(_dune_target_gmp_backend BACKEND ${ADD_GMP_BACKEND})

  if(_dune_target_gmp_backend STREQUAL "MPFR")
    foreach(_target ${ADD_GMP_TARGETS})
      target_link_libraries(${_target} PUBLIC MPFR::mpreal)
      target_compile_definitions(${_target} PUBLIC HAVE_GMP=1 HAVE_MPFR=1)
    endforeach()
  elseif(_dune_target_gmp_backend STREQUAL "GMPXX")
    foreach(_target ${ADD_GMP_TARGETS})
      target_link_libraries(${_target} PUBLIC GMP::gmpxx)
      target_compile_definitions(${_target} PUBLIC HAVE_GMP=1 HAVE_GMPXX=1)
    endforeach()
  endif()
endfunction(add_dune_gmp_flags)
