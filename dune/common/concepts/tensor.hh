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

/**
 * \brief An `Extents` is a description of a multi-dimensional index-space.
 *
 * `Extents` defined the index-space in terms of extents `.extent(i)` per dimension `i`.
 * The total number of dimensions is given by the `::rank()` of the index-space. In case the
 * extent of a dimension is statically known, it is returned by the `::static_extent(i)`, otherwise
 * this function returns a `std::dynamic_extent` constant.
 */
template <class E>
concept Extents = requires(E e, typename E::rank_type r)
{
  { E::rank() } -> std::convertible_to<typename E::rank_type>;
  { E::rank_dynamic() } -> std::convertible_to<typename E::rank_type>;
  { E::static_extent(r) } -> std::convertible_to<std::size_t>;
  { e.extent(r) } -> std::convertible_to<typename E::index_type>;
};


/**
 * \brief A `Tensor` is a multi-dimensional container with given extents.
 *
 * It is required that a `Tensor` provides its extents by the member `.extents()` as an object
 * that models the `Extents` concept, and that the tensor can be accessed by `operator[]` with
 * a multi-index passed as a `std::array` of indices. The number of indices is equal to the rank
 * of the tensor. This access is assumed to be valid of all indices in the index-space defined
 * by the tensor extents.
 *
 * \b Examples:
 * - `Dune::Std::mdarray`
 * - `Dune::Std::mdspan`
 * - `Dune::Tensor`
 * - `Dune::TensorSpan`
 */
template <class T>
concept Tensor = Extents<typename T::extents_type> &&
requires(T tensor, std::array<typename T::index_type, T::rank()> indices)
{
  { tensor.extents() } -> std::convertible_to<typename T::extents_type>;
  tensor[indices];
};

//! A `TensorWithRank` is a `Tensor` with given tensor-rank `rank`.
template <class T, std::size_t rank>
concept TensorWithRank = Tensor<T> && T::rank() == rank;

//! A `Vector` is a `Tensor` of rank 1.
template <class T>
concept Vector = TensorWithRank<T,1>;

//! A `Matrix` is a `Tensor` of rank 2.
template <class T>
concept Matrix = TensorWithRank<T,2>;


static_assert(Concept::TensorWithRank<Archetypes::Tensor<double,0>,0>);
static_assert(Concept::Vector<Archetypes::Tensor<double,1>>);
static_assert(Concept::Matrix<Archetypes::Tensor<double,2>>);

} // end namespace Dune::Concept

#endif // DUNE_COMMON_CONCEPTS_TENSOR_HH
