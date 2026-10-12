// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_CONCEPTS_MULTIDIMENSIONAL_HH
#define DUNE_COMMON_CONCEPTS_MULTIDIMENSIONAL_HH

#include <array>
#include <concepts>
#include <cstddef>

namespace Dune::Concept {

/**
 * \brief An `Extents` type describes a multi-dimensional index space.
 * \ingroup CxxConcepts
 *
 * The index space is the cartesian product `[0,e.extent(0)) x ... x [0,e.extent(rank-1))`.
 * The number of dimensions is given by the static `E::rank()`. If the extent of a
 * dimension is known at compile time, it is returned by `E::static_extent(r)`, otherwise
 * this function returns `std::dynamic_extent`.
 *
 * \par Notation:
 * - `e`: a const object of type `E`
 * - `r`: an object of type `E::rank_type`
 *
 * \par Valid Expressions:
 * - `E::rank()`: the number of dimensions of the index space
 * - `E::rank_dynamic()`: the number of dimensions with dynamic extent
 * - `E::static_extent(r)`: the static extent of dimension `r`, or `std::dynamic_extent`
 * - `e.extent(r)`: the extent of dimension `r`
 *
 * \par Models:
 * - `Dune::Std::extents`
 * - `std::extents`
 *
 * \experimental
 * \hideinitializer
 **/
template <class E>
concept Extents = std::copyable<E> &&
requires(const E e, typename E::rank_type r)
{
  requires std::integral<typename E::index_type>;
  requires std::integral<typename E::size_type>;
  requires std::integral<typename E::rank_type>;
  { E::rank() } -> std::convertible_to<typename E::rank_type>;
  { E::rank_dynamic() } -> std::convertible_to<typename E::rank_type>;
  { E::static_extent(r) } -> std::convertible_to<std::size_t>;
  { e.extent(r) } -> std::convertible_to<typename E::index_type>;
};


/**
 * \brief A `MultiDimensional` object provides element access on a multi-dimensional
 * index space, following the interface of `mdspan`.
 * \ingroup CxxConcepts
 *
 * The index space is described by the `extents()` of the object, a model of \ref Extents.
 * The elements are accessed by `operator[]` with a multi-index passed as `std::array`
 * of `rank()` indices. This access is valid for all multi-indices in the index space.
 *
 * Generic code constrained by this concept should only use the interface described here.
 * Some models provide more methods, but with a different meaning. For example,
 * `FieldMatrix::size()` returns the number of rows, while `mdspan::size()` returns
 * the total number of elements, and `FieldMatrix` provides an iterator over its rows
 * and `m[i][j]` element access, which `mdspan` does not provide.
 *
 * \par Notation:
 * - `t`: a const object of type `T`
 * - `r`: an object of type `T::rank_type`
 * - `indices`: a `std::array<T::index_type, T::rank()>`
 *
 * \par Valid Expressions:
 * - `T::rank()`, `T::rank_dynamic()`, `T::static_extent(r)`: properties of the index space
 * - `t.extent(r)`: the extent of dimension `r`
 * - `t.extents()`: the index space of type `T::extents_type`
 * - `t[indices]`: access to the element at the multi-index `indices`
 *
 * \par Models:
 * - `Dune::Std::mdspan`, `std::mdspan`
 * - `Dune::Std::mdarray`
 * - `Dune::FieldVector`, `Dune::DynamicVector`
 * - `Dune::FieldMatrix`, `Dune::DynamicMatrix`
 *
 * \note The concept requires the type `T` without reference or cv-qualifiers.
 *
 * \experimental
 * \hideinitializer
 **/
template <class T>
concept MultiDimensional = Extents<typename T::extents_type> &&
requires(const T t, typename T::rank_type r,
         const std::array<typename T::index_type, T::extents_type::rank()>& indices)
{
  typename T::value_type;
  requires std::same_as<typename T::index_type, typename T::extents_type::index_type>;
  requires std::same_as<typename T::rank_type, typename T::extents_type::rank_type>;
  requires (T::rank() == T::extents_type::rank());
  { T::rank() } -> std::convertible_to<typename T::rank_type>;
  { T::rank_dynamic() } -> std::convertible_to<typename T::rank_type>;
  { T::static_extent(r) } -> std::convertible_to<std::size_t>;
  { t.extent(r) } -> std::convertible_to<typename T::index_type>;
  { t.extents() } -> std::convertible_to<typename T::extents_type>;
  t[indices];
};


/**
 * \brief A \ref MultiDimensional object with a given rank.
 * \ingroup CxxConcepts
 *
 * \experimental
 * \hideinitializer
 **/
template <class T, std::size_t rank>
concept MultiDimensionalWithRank = MultiDimensional<T> && (T::rank() == rank);

} // end namespace Dune::Concept

#endif // DUNE_COMMON_CONCEPTS_MULTIDIMENSIONAL_HH
