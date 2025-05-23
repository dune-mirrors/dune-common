cmake_minimum_required(VERSION 3.16)
project(dune-common-benchmark LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

# find a build or installed directory with dune-common
find_package(dune-common REQUIRED)
include(DuneMacros)

find_package(benchmark)

link_libraries(Dune::Common)

dune_add_test(SOURCES tensordotbenchmark.cc
              LINK_LIBRARIES benchmark::benchmark
              CMAKE_GUARD benchmark_FOUND)
