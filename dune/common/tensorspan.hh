// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TENSORSPAN_HH
#define DUNE_COMMON_TENSORSPAN_HH

#include <array>
#include <concepts>
#include <type_traits>

#include <dune/common/ftraits.hh>
#include <dune/common/tensormixin.hh>
#include <dune/common/std/default_accessor.hh>
#include <dune/common/std/extents.hh>
#include <dune/common/std/layout_right.hh>
#include <dune/common/std/mdspan.hh>

namespace Dune {

/**
 * \brief A multi-dimensional span with tensor interface.
 * \nosubgrouping
 *
 * \tparam Element  The element type stored in the tensor.
 * \tparam Extents  The type representing the tensor extents. Must be compatible
 *                  to `Dune::Std::extents`.
 * \tparam Layout   A class providing the index mapping from the multi-dimensional
 *                  index space to a single index usable in the `Container`. See
 *                  the `Dune::Concept::Layout` concept.
 * \tparam Accessor A class that defines types and operations to create a reference
 *                  to a single element stored by a data pointer and index.
 **/
template <class Element, class Extents,
          class Layout = Std::layout_right,
          class Accessor= Std::default_accessor<Element>>
class TensorSpan
    : public TensorMixin<TensorSpan<Element,Extents,Layout,Accessor>,
        Std::mdspan<Element,Extents,Layout,Accessor>>
{
  using self_type = TensorSpan;
  using storage_type = Std::mdspan<Element,Extents,Layout,Accessor>;
  using base_type = TensorMixin<self_type,storage_type>;

public:
  using element_type =	Element;
  using extents_type = Extents;
  using layout_type = Layout;
  using accessor_type = Accessor;
  using mapping_type = typename base_type::mapping_type;

public:
  /// \name TensorSpan constructors
  /// @{

  using base_type::base_type;

  /// \brief Converting constructor
  template <class V, class E, class L, class A, class M = typename L::template mapping<E>>
    requires (std::is_constructible_v<mapping_type, const M&> &&
              std::is_constructible_v<accessor_type, const A&>)
  constexpr TensorSpan (const TensorSpan<V,E,L,A>& other)
    : base_type{other}
  {}

  /// \brief Converting move constructor
  template <class V, class E, class L, class A, class M = typename L::template mapping<E>>
    requires (std::is_constructible_v<mapping_type, const M&> &&
              std::is_constructible_v<accessor_type, const A&>)
  constexpr TensorSpan (TensorSpan<V,E,L,A>&& tensor)
    : base_type{std::move(tensor)}
  {}

  /// \brief base copy constructor
  constexpr TensorSpan (const storage_type& tensor)
    : base_type{tensor}
  {}

  /// \brief base move constructor
  constexpr TensorSpan (storage_type&& tensor)
    : base_type{std::move(tensor)}
  {}

  /// @}
};

// deduction guides
// @{

template <class CArray>
  requires (std::is_array_v<CArray> && (std::rank_v<CArray> == 1))
TensorSpan (CArray&)
  -> TensorSpan<std::remove_all_extents_t<CArray>, Std::extents<std::size_t, std::extent_v<CArray,0>>>;

template <class Pointer>
  requires (std::is_pointer_v<std::remove_reference_t<Pointer>>)
TensorSpan (Pointer&&)
  -> TensorSpan<std::remove_pointer_t<std::remove_reference_t<Pointer>>, Std::extents<std::size_t>>;

template <class element_type, std::convertible_to<std::size_t>... II>
  requires (sizeof...(II) > 0)
TensorSpan (element_type*, II...)
  -> TensorSpan<element_type, Std::dextents<std::size_t, sizeof...(II)>>;

template <class element_type, std::integral SizeType, std::size_t N>
TensorSpan (element_type*, std::span<SizeType,N>&)
  -> TensorSpan<element_type, Std::dextents<std::size_t, N>>;

template <class element_type, std::integral SizeType, std::size_t N>
TensorSpan (element_type*, const std::array<SizeType,N>&)
  -> TensorSpan<element_type, Std::dextents<std::size_t, N>>;

template <class element_type, std::integral IndexType, std::size_t... exts>
TensorSpan (element_type*, const Std::extents<IndexType,exts...>&)
  -> TensorSpan<element_type, Std::extents<IndexType,exts...>>;

template <class element_type, class Mapping,
  class Extents = typename Mapping::extents_type,
  class Layout = typename Mapping::layout_type>
TensorSpan (element_type*, const Mapping&)
  -> TensorSpan<element_type, Extents, Layout>;

template <class Mapping, class Accessor,
  class DataHandle = typename Accessor::data_handle_type,
  class Element = typename Accessor::element_type,
  class Extents = typename Mapping::extents_type,
  class Layout = typename Mapping::layout_type>
TensorSpan (const DataHandle&, const Mapping&, const Accessor&)
  -> TensorSpan<Element, Extents, Layout, Accessor>;

template <class V, class E, class L, class A>
TensorSpan (Std::mdspan<V,E,L,A>)
  -> TensorSpan<V,E,L,A>;

/// @}


template <class Element, class Extents, class Layout, class Accessor>
struct FieldTraits< TensorSpan<Element,Extents,Layout,Accessor> >
{
  using field_type = typename FieldTraits<Element>::field_type;
  using real_type = typename FieldTraits<Element>::real_type;
};

} // end namespace Dune

#endif // DUNE_COMMON_TENSORSPAN_HH
