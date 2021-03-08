# The DUNE way to compile MPI applications is to use the CXX
# compiler with MPI flags usually used for C. CXX bindings
# are deactivated to prevent ABI problems.
#
# .. cmake_function:: add_dune_mpi_flags
#
#    .. cmake_param:: targets
#       :single:
#       :required:
#       :positional:
#
#       The target list to add the MPI flags to.
#

# text for feature summary
set_package_properties("MPI" PROPERTIES
  DESCRIPTION "Message Passing Interface library"
  PURPOSE "Parallel programming on multiple processors")

if(MPI_C_FOUND)
  set(HAVE_MPI ${MPI_C_FOUND})

  # provide flags to manually disable c++ bindings in MPI_C-library
  set(MPI_DISABLE_CXX_BINDINGS MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX MPIPP_H MPI_NO_CPPBIND)

  # TODO check on where to position this exactly, doesn't look completely thought through
  dune_register_package_flags(COMPILE_DEFINITIONS "ENABLE_MPI=1;${MPI_DISABLE_CXX_BINDINGS}"
                              LIBRARIES MPI::MPI_C)
endif()

# adds MPI flags to the targets
function(add_dune_mpi_flags)
  cmake_parse_arguments(ADD_MPI "SOURCE_ONLY;OBJECT" "" "" ${ARGN})
  set(targets ${ADD_MPI_UNPARSED_ARGUMENTS})

  if(MPI_C_FOUND)
    foreach(target ${targets})
      target_link_libraries(${target} PUBLIC MPI::MPI_C)
      target_compile_definitions(${target} PUBLIC "ENABLE_MPI=1;${MPI_DISABLE_CXX_BINDINGS}")
    endforeach(target)
  endif()
endfunction(add_dune_mpi_flags)
