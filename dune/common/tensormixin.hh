// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TENSORMIXIN_HH
#define DUNE_COMMON_TENSORMIXIN_HH

#include <cassert>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <dune/common/boundschecking.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/indices.hh>
#include <dune/common/concepts/number.hh>
#include <dune/common/std/type_traits.hh>

namespace Dune {

/**
 * \brief A tensor interface-class providing common functionality.
 * \nosubgrouping
 *
 * The CRTP interface mixin-class for tensors extending the mdarray and mdspan implementations
 * by common functionality for element access and size information and mathematical
 * operations.
 *
 * \tparam Derived  The tensor type derived from this class.
 * \tparam Base     Either mdarray or mdspan base type for storage.
 **/
template <class Derived, class Base>
class TensorMixin
    : public Base
{
  using self_type = TensorMixin;
  using derived_type = Derived;
  using base_type = Base;

  template <class B>
  using const_reference_t = typename B::const_reference;

public:
  using element_type = typename base_type::element_type;
  using value_type = std::remove_const_t<element_type>;
  using field_type = typename FieldTraits<value_type>::field_type;
  using extents_type = typename base_type::extents_type;
  using index_type = typename base_type::index_type;
  using layout_type = typename base_type::layout_type;
  using mapping_type = typename base_type::mapping_type;
  using reference = typename base_type::reference;
  using const_reference = Std::detected_or_t<reference,const_reference_t,base_type>;

protected:
  /// Derive constructors from base class
  using base_type::base_type;


  /// \brief base copy constructor
  constexpr TensorMixin (const base_type& tensor)
        noexcept(std::is_nothrow_copy_constructible_v<base_type>)
    : base_type{tensor}
  {}

  /// \brief base move constructor
  constexpr TensorMixin (base_type&& tensor)
        noexcept(std::is_nothrow_move_constructible_v<base_type>)
    : base_type{std::move(tensor)}
  {}

public:
  /// \name Multi index access
  /// @{

  /// \brief Access specified element at position (i0,i1,...) with mutable access
  template <std::convertible_to<index_type>... Indices>
    requires (sizeof...(Indices) == extents_type::rank())
  constexpr reference operator() (Indices... indices)
  {
    DUNE_ASSERT_BOUNDS(indexInIndexSpace(indices...));
    return base_type::operator[](std::array<index_type,extents_type::rank()>{index_type(indices)...});
  }

  /// \brief Access specified element at position (i0,i1,...) with const access
  template <std::convertible_to<index_type>... Indices>
    requires (sizeof...(Indices) == extents_type::rank())
  constexpr const_reference operator() (Indices... indices) const
  {
    DUNE_ASSERT_BOUNDS(indexInIndexSpace(indices...));
    return base_type::operator[](std::array<index_type,extents_type::rank()>{index_type(indices)...});
  }

  /**
   * \brief Access specified element at position (i0,i1,...) with mutable access
   * \throws Dune::RangeError if the indices are out of the index space `[0,extent_0)x...x[0,extent_{r-1})`.
   */
  template <std::convertible_to<index_type>... Indices>
    requires (sizeof...(Indices) == extents_type::rank())
  constexpr reference at (Indices... indices)
  {
    if (not indexInIndexSpace(indices...))
      DUNE_THROW(Dune::RangeError, "Indices out of bounds.");
    return base_type::operator[](std::array<index_type,extents_type::rank()>{index_type(indices)...});
  }

  /**
   * \brief Access specified element at position (i0,i1,...) with const access
   * \throws Dune::RangeError if the indices are out of the index space `[0,extent_0)x...x[0,extent_{r-1})`.
   */
  template <std::convertible_to<index_type>... Indices>
    requires (sizeof...(Indices) == extents_type::rank())
  constexpr const_reference at (Indices... indices) const
  {
    if (not indexInIndexSpace(indices...))
      DUNE_THROW(Dune::RangeError, "Indices out of bounds.");
    return base_type::operator[](std::array<index_type,extents_type::rank()>{index_type(indices)...});
  }

  /// \brief Access element at position [{i0,i1,...}]
  template <std::convertible_to<index_type> Index>
  constexpr reference operator[] (const std::array<Index,extents_type::rank()>& indices)
  {
    DUNE_ASSERT_BOUNDS(indexInIndexSpace(indices));
    return base_type::operator[](indices);
  }

  /// \brief Access element at position [{i0,i1,...}]
  template <std::convertible_to<index_type> Index>
  constexpr const_reference operator[] (const std::array<Index,extents_type::rank()>& indices) const
  {
    DUNE_ASSERT_BOUNDS(indexInIndexSpace(indices));
    return base_type::operator[](indices);
  }

  /// \brief Access vector-element at position [i0] with mutable access.
  constexpr reference operator[] (index_type index)
        requires (extents_type::rank() == 1)
  {
    DUNE_ASSERT_BOUNDS(indexInIndexSpace(index));
    return base_type::operator[](std::array{index});
  }

  /// \brief Access vector-element at position [i0] with const access.
  constexpr const_reference operator[] (index_type index) const
        requires (extents_type::rank() == 1)
  {
    DUNE_ASSERT_BOUNDS(indexInIndexSpace(index));
    return base_type::operator[](std::array{index});
  }

  /**
   * \brief Return true when (i0,i1,...) is in pattern.
   * This is always true for dense tensors.
   **/
  template <std::convertible_to<index_type>... Indices>
    requires (sizeof...(Indices) == extents_type::rank())
  constexpr bool exists (Indices... indices) const noexcept
  {
    return indexInIndexSpace(indices...);
  }

  /// @}


  /// \name Size information
  /// @{

  /// \brief Number of rows of a rank-2 tensor
  constexpr index_type rows () const noexcept
        requires (extents_type::rank() == 2)
  {
    return asBase().extent(0);
  }

  /// \brief Number of columns of a rank-2 tensor
  constexpr index_type cols () const noexcept
        requires (extents_type::rank() == 2)
  {
    return asBase().extent(1);
  }

  /// @}


  /// \name Conversion to the underlying value if rank is zero
  // @{

  constexpr operator reference () noexcept
        requires (extents_type::rank() == 0)
  {
    return base_type::operator[](std::array<index_type,0>{});
  }

  constexpr operator const_reference () const noexcept
        requires (extents_type::rank() == 0)
  {
    return base_type::operator[](std::array<index_type,0>{});
  }

  /// @}


  /// \brief Comparison of two TensorMixins for equality
  friend constexpr bool operator== (const TensorMixin& lhs, const TensorMixin& rhs) noexcept
  {
    return static_cast<const base_type&>(lhs) == static_cast<const base_type&>(rhs);
  }

private:
  // Check whether a tuple of indices is in the index space `[0,extent_0)x...x[0,extent_{r-1})`.
  template <class... Indices>
  [[nodiscard]] constexpr bool indexInIndexSpace (Indices... indices) const noexcept
  {
    return unpackIntegerSequence([&](auto... i) {
      return ( (0 <= indices && index_type(indices) < asBase().extent(i)) && ... );
    }, std::make_index_sequence<sizeof...(Indices)>{});
  }

  // Check whether a tuple of indices is in the index space `[0,extent_0)x...x[0,extent_{r-1})`.
  template <class Index>
  [[nodiscard]] constexpr bool indexInIndexSpace (const std::array<Index,extents_type::rank()>& indices) const noexcept
  {
    return unpackIntegerSequence([&](auto... i) {
      return ( (0 <= indices[i] && index_type(indices[i]) < asBase().extent(i)) && ... );
    }, std::make_index_sequence<extents_type::rank()>{});
  }

private:

  derived_type const& asDerived () const
  {
    return static_cast<derived_type const&>(*this);
  }

  derived_type& asDerived ()
  {
    return static_cast<derived_type&>(*this);
  }

  base_type const& asBase () const
  {
    return static_cast<base_type const&>(*this);
  }

  base_type& asBase ()
  {
    return static_cast<base_type&>(*this);
  }
};

// specialization for rank-0 tensor and comparison with scalar
template <class D, class B, Concept::Number S>
  requires (B::extents_type::rank() == 0)
constexpr bool operator== (const TensorMixin<D,B>& lhs, const S& number) noexcept
{
  return lhs() == number;
}

// specialization for rank-0 tensor and comparison with scalar
template <Concept::Number S, class D, class B>
  requires (B::extents_type::rank() == 0)
constexpr bool operator== (const S& number, const TensorMixin<D,B>& rhs) noexcept
{
  return number == rhs();
}

template <class D, class B>
struct FieldTraits< TensorMixin<D,B> >
{
  using field_type = typename FieldTraits<D>::field_type;
  using real_type = typename FieldTraits<D>::real_type;
};

} // end namespace Dune

#endif // DUNE_COMMON_TENSORMIXIN_HH
