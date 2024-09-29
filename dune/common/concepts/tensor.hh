// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_CONCEPTS_TENSOR_HH
#define DUNE_COMMON_CONCEPTS_TENSOR_HH

#include <array>
#include <concepts>

#include <dune/common/concepts/archetypes/tensor.hh>

namespace Dune::Concept {

template <class T>
concept Tensor = requires(T tensor, typename T::rank_type r)
{
  { T::rank() } -> std::convertible_to<std::size_t>;
  { T::static_extent(r) } -> std::same_as<std::size_t>;
  { tensor.extent(r) } -> std::convertible_to<typename T::index_type>;
}
&& requires(T tensor, std::array<typename T::index_type, T::rank()> indices)
{
  { tensor[indices] } -> std::convertible_to<typename T::reference>;
};

static_assert(Concept::Tensor<Archetypes::Tensor<double,0>>);
static_assert(Concept::Tensor<Archetypes::Tensor<double,1>>);
static_assert(Concept::Tensor<Archetypes::Tensor<double,2>>);

} // end namespace Dune::Concept

#endif // DUNE_COMMON_CONCEPTS_TENSOR_HH
