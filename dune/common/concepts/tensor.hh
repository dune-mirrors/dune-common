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

template <class E, std::size_t rank = E::rank()>
concept Extents = requires(E extents, std::size_t i)
{
  { E::rank() } -> std::convertible_to<std::size_t>;
  { E::static_extent(i) } -> std::same_as<std::size_t>;
  { extents.extent(i) } -> std::convertible_to<typename E::index_type>;
} && E::rank() == rank;

template <class T, std::size_t rank = T::rank()>
concept Tensor = Extents<T,rank> && requires(T tensor)
{
  requires Extents<typename T::extents_type,rank>;
  { tensor.extents() } -> std::convertible_to<typename T::extents_type>;
};

template <class T>
concept Vector = Tensor<T,1>;

template <class T>
concept Matrix = Tensor<T,2>;

static_assert(Concept::Tensor<Archetypes::Tensor<double,0>,0>);
static_assert(Concept::Vector<Archetypes::Tensor<double,1>>);
static_assert(Concept::Matrix<Archetypes::Tensor<double,2>>);


template <class T, std::size_t rank = T::rank()>
concept RandomAccessTensor = Tensor<T,rank> &&
requires(T tensor, std::array<typename T::index_type, rank> indices)
{
  tensor[indices];
};

template <class T>
concept RandomAccessVector = RandomAccessTensor<T,1>;

template <class T>
concept RandomAccessMatrix = RandomAccessTensor<T,2>;

static_assert(Concept::RandomAccessTensor<Archetypes::Tensor<double,0>,0>);
static_assert(Concept::RandomAccessVector<Archetypes::Tensor<double,1>>);
static_assert(Concept::RandomAccessMatrix<Archetypes::Tensor<double,2>>);

} // end namespace Dune::Concept

#endif // DUNE_COMMON_CONCEPTS_TENSOR_HH
