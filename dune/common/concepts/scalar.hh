// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_CONCEPTS_SCALAR_HH
#define DUNE_COMMON_CONCEPTS_SCALAR_HH

// check whether c++20 concept can be used
#if __has_include(<version>) && __has_include(<concepts>)
  #include <version>
  #if  __cpp_concepts >= 201907L && __cpp_lib_concepts >= 202002L
    #ifndef DUNE_ENABLE_CONCEPTS
    #define DUNE_ENABLE_CONCEPTS 1
    #endif
  #endif
#endif

#if DUNE_ENABLE_CONCEPTS

#include <complex>
#include <dune/common/typetraits.hh>

namespace Dune::Concept {

template <class N>
concept Scalar = Dune::IsNumber<N>::value;

static_assert(Scalar<short>);
static_assert(Scalar<unsigned short>);
static_assert(Scalar<int>);
static_assert(Scalar<unsigned int>);
static_assert(Scalar<long>);
static_assert(Scalar<unsigned long>);

static_assert(Scalar<float>);
static_assert(Scalar<double>);
static_assert(Scalar<long double>);

static_assert(Scalar<std::complex<float>>);
static_assert(Scalar<std::complex<double>>);
static_assert(Scalar<std::complex<long double>>);

} // end namespace Dune::Concept

#endif // DUNE_ENABLE_CONCEPTS

#endif // DUNE_COMMON_CONCEPTS_SCALAR_HH
