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

template <class E>
concept Extents = requires(E extents, std::size_t i)
{
  { E::rank() } -> std::convertible_to<std::size_t>;
  { E::static_extent(i) } -> std::same_as<std::size_t>;
  { extents.extent(i) } -> std::convertible_to<typename E::index_type>;
};

template <class T>
concept Tensor = Extents<T> && requires(T tensor)
{
  requires Extents<typename T::extents_type>;
  { tensor.extents() } -> std::convertible_to<typename T::extents_type>;
};

static_assert(Concept::Tensor<Archetypes::Tensor<double,0>>);
static_assert(Concept::Tensor<Archetypes::Tensor<double,1>>);
static_assert(Concept::Tensor<Archetypes::Tensor<double,2>>);


template <class T>
concept RandomAccessTensor = Tensor<T> &&
requires(T tensor, std::array<typename T::index_type, T::rank()> indices)
{
  tensor[indices];
};

static_assert(Concept::RandomAccessTensor<Archetypes::Tensor<double,0>>);
static_assert(Concept::RandomAccessTensor<Archetypes::Tensor<double,1>>);
static_assert(Concept::RandomAccessTensor<Archetypes::Tensor<double,2>>);

} // end namespace Dune::Concept

#endif // DUNE_COMMON_CONCEPTS_TENSOR_HH
