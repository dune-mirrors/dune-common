// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_CONCEPTS_SCALAR_HH
#define DUNE_COMMON_CONCEPTS_SCALAR_HH

#include <complex>

#include <dune/common/typetraits.hh>

namespace Dune::Concept {

/**
 * \brief Whether this type acts as a scalar in the context of
 *        (hierarchically blocked) containers
 */
template <class N>
concept Scalar = Dune::IsNumber<N>::value;

} // end namespace Dune::Concept

#endif // DUNE_COMMON_CONCEPTS_SCALAR_HH
