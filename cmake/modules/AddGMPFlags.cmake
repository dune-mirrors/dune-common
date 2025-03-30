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

# set HAVE_GMP for the config.h file
if(GMP_FOUND OR MPFR_FOUND)
  set(HAVE_GMP TRUE)
else()
  set(HAVE_GMP FALSE)
endif()

# register all GMP related flags
if(MPFR_FOUND)
  dune_register_package_flags(
    LIBRARIES MPFR::mpreal
    COMPILE_DEFINITIONS "HAVE_GMP=1;HAVE_MPFR=1"
  )
elseif(GMP_FOUND)
  message(WARNING "The library GMPXX is outdated and deprecated. Use MPFR C++ (mpreal)  instead.")
  dune_register_package_flags(
    LIBRARIES GMP::gmpxx
    COMPILE_DEFINITIONS "HAVE_GMP=1;HAVE_GMPXX=1"
  )
endif()

# add function to link against the GMP library
function(add_dune_gmp_flags _targets)
  if(MPFR_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} PUBLIC MPFR::mpreal)
      target_compile_definitions(${_target} PUBLIC HAVE_GMP=1 HAVE_MPFR=1)
    endforeach(_target ${_targets})
  elseif(GMP_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} PUBLIC GMP::gmpxx)
      target_compile_definitions(${_target} PUBLIC HAVE_GMP=1 HAVE_GMPXX=1)
    endforeach(_target ${_targets})
  endif()
endfunction(add_dune_gmp_flags)
