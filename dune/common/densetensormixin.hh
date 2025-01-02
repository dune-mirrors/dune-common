// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_DENSETENSORMIXIN_HH
#define DUNE_COMMON_DENSETENSORMIXIN_HH

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include <dune/common/boundschecking.hh>
#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/indices.hh>
#include <dune/common/math.hh>
#include <dune/common/tensordot.hh>
#include <dune/common/concepts/number.hh>
#include <dune/common/concepts/tensor.hh>
#include <dune/common/std/type_traits.hh>
#include <dune/common/test/foreachindex.hh>

namespace Dune {
namespace Impl {

// Take the extents from both index spaces and use as many static extents as possible
template <class I1, std::size_t... exts1, class I2, std::size_t... exts2>
  requires (sizeof...(exts1) == sizeof...(exts2))
constexpr auto combinedExtents (const Std::extents<I1,exts1...>& e1,
                                const Std::extents<I2,exts2...>& e2)
{
  using I = std::common_type_t<I1,I2>;
  return [&]<std::size_t... II>(std::index_sequence<II...>) {
    return Std::extents<I, (exts1 == Std::dynamic_extent ? exts2 : exts1)...>{I(e1.extent(II))...};
  }(std::make_index_sequence<sizeof...(exts1)>{});
}

} // end namespace Impl

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
class DenseTensorMixin
    : public Base
{
  using self_type = DenseTensorMixin;
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

public:
  /// \brief Assign `value` to each component of the tensor
  template <Concept::Number S>
    requires (std::is_assignable_v<value_type&, const S&>)
  constexpr derived_type& operator= (const S& value)
      noexcept(std::is_nothrow_assignable_v<value_type&, const S&>)
  {
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] = value;
    });
    return asDerived();
  }

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


  /// \name Elementwise operations
  // @{

  /// \brief Vector space operation ( *this += x )
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr derived_type& operator+= (const T& x)
  {
    assert(base_type::extents() == x.extents());
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] += x[index];
    });
    return asDerived();
  }

  /// \brief Binary elementwise addition of two tensors
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr Concept::RandomAccessTensor auto operator+ (const T& x) const
  {
    assert(base_type::extents() == x.extents());
    using I1 = std::array<index_type,extents_type::rank()>;
    using I2 = std::array<typename T::index_type,T::rank()>;
    using V = std::decay_t<decltype((*this)[std::declval<I1>()] + x[std::declval<I2>()])>;

    // create a copy of *this with as many static extents as possible
    auto result = [&]<class I, std::size_t... exts>(const Std::extents<I,exts...>&) {
      return Tensor<V,exts...>{*this};
    }(Impl::combinedExtents(base_type::extents(), x.extents()));

    result += x;
    return result;
  }

  /// \brief Vector space operation ( *this -= x )
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr derived_type& operator-= (const T& x)
  {
    assert(base_type::extents() == x.extents());
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] -= x[index];
    });
    return asDerived();
  }

  /// \brief Binary elementwise subtraction of two tensors
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr Concept::RandomAccessTensor auto operator- (const T& x) const
  {
    assert(base_type::extents() == x.extents());
    using I1 = std::array<index_type,extents_type::rank()>;
    using I2 = std::array<typename T::index_type,T::rank()>;
    using V = std::decay_t<decltype((*this)[std::declval<I1>()] - x[std::declval<I2>()])>;

    // create a copy of *this with as many static extents as possible
    auto result = [&]<class I, std::size_t... exts>(const Std::extents<I,exts...>&) {
      return Tensor<V,exts...>{*this};
    }(Impl::combinedExtents(base_type::extents(), x.extents()));

    result -= x;
    return result;
  }

  /// \brief Elementwise negation of the tensor
  constexpr Concept::RandomAccessTensor auto operator- () const
  {
    // create a copy of *this
    auto result = Tensor{base_type::extents(), value_type(0)};
    forEachIndex(base_type::extents(), [&](auto&& index) {
      result[index] = -(*this)[index];
    });
    return result;
  }

  /// \brief Vector space axpy operation ( *this += alpha x )
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr derived_type& axpy (const field_type& alpha, const T& x)
  {
    assert(base_type::extents() == x.extents());
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] += alpha * x[index];
    });
    return asDerived();
  }

  /// \brief Vector space aypx operation ( *this = alpha * (*this) + x )
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr derived_type& aypx (const field_type& alpha, const T& x)
  {
    assert(base_type::extents() == x.extents());
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] = alpha * (*this)[index] + x[index];
    });
    return asDerived();
  }

  /// \brief Vector space operation ( *this *= scalar )
  template <Concept::Number S>
  constexpr derived_type& operator*= (const S& scalar)
  {
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] *= scalar;
    });
    return asDerived();
  }

  /// \brief Elementwise scalar multiplication of the tensor
  template <Concept::Number S>
  constexpr friend Concept::RandomAccessTensor auto operator* (const self_type& self, const S& scalar)
  {
    auto result = Tensor{self};
    result *= scalar;
    return result;
  }

  /// \brief Elementwise scalar multiplication of the tensor
  template <Concept::Number S>
  constexpr friend Concept::RandomAccessTensor auto operator* (const S& scalar, const self_type& self)
  {
    auto result = Tensor{self};
    result *= scalar;
    return result;
  }

  /// \brief Vector space operation ( *this /= scalar )
  template <Concept::Number S>
  constexpr derived_type& operator/= (const S& scalar)
  {
    forEachIndex(base_type::extents(), [&](auto&& index) {
      (*this)[index] /= scalar;
    });
    return asDerived();
  }

  /// \brief Elementwise scalar division of the tensor
  template <Concept::Number S>
  constexpr friend Concept::RandomAccessTensor auto operator/ (const self_type& self, const S& scalar)
  {
    auto result = Tensor{self};
    result /= scalar;
    return result;
  }

  /// @}


  /// \name tensor dot-products
  // @{

  /// \brief Returns the tensor product with contraction over a single index `A_{ij} B_{jkl}`
  template <Concept::RandomAccessTensor T>
    requires (extents_type::rank() >= 1 && T::rank() >= 1 &&
      Impl::checkStaticExtents<1, extents_type, typename T::extents_type>())
  constexpr Concept::RandomAccessTensor auto operator* (const T& tensor) const
  {
    return tensordot<1>(*this, tensor);
  }

  /// \brief Returns the Hermitian tensor product with contraction over a single index `conj(A_{ij}) B_{jkl}`
  template <Concept::RandomAccessTensor T>
    requires (extents_type::rank() >= 1 && T::rank() >= 1 &&
      Impl::checkStaticExtents<1, extents_type, typename T::extents_type>())
  constexpr Concept::RandomAccessTensor auto dot (const T& tensor) const
  {
    return tensordot(*this, tensor, Indices::_1, std::plus<>{}, DotProduct{});
  }

  /// \brief Returns the Hermitian tensor product with contraction over two indices `conj(A_{ijk}) B_{jkl}`
  template <Concept::RandomAccessTensor T>
    requires (extents_type::rank() >= 2 && T::rank() >= 2 &&
      Impl::checkStaticExtents<1, extents_type, typename T::extents_type>())
  constexpr Concept::RandomAccessTensor auto ddot (const T& tensor) const
  {
    return tensordot(*this, tensor, Indices::_2, std::plus<>{}, DotProduct{});
  }


  /// \brief y = A x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void mv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    y = 0;
    tensordotOut<1>(*this,x,y);
  }

  /// \brief y = A^T x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void mtv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    y = 0;
    tensordotOut<1>(x,*this,y);
  }

  /// \brief y = A^H x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void mhv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    y = 0;
    tensordotOut(x,*this,y, Indices::_1,
      std::plus<>{}, [](auto&& a, auto&& b) { return a * conjugateComplex(b); });
  }

  /// \brief y += A x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void umv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut<1>(*this,x,y);
  }

  /// \brief y -= A x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void mmv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(*this,x,y, Indices::_1, std::minus<>{}, std::multiplies<>{});
  }

  /// \brief y += A^T x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void umtv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut<1>(x,*this,y);
  }

  /// \brief y -= A^T x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void mmtv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(x,*this,y, Indices::_1, std::minus<>{}, std::multiplies<>{});
  }

  /// \brief y += A^H x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void umhv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(x,*this,y, Indices::_1,
      std::plus<>{}, [](auto&& a, auto&& b) { return a * conjugateComplex(b); });
  }

  /// \brief y -= A^H x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void mmhv (const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(x,*this,y, Indices::_1,
      std::minus<>{}, [](auto&& a, auto&& b) { return a * conjugateComplex(b); });
  }

  /// \brief y += alpha A x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void usmv (const typename FieldTraits<VectorOut>::field_type& alpha,
                       const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(*this,x, y, Indices::_1,
      std::plus<>{}, [alpha](auto&& a, auto&& b) { return alpha * a * b; });
  }

  /// \brief y += alpha A^T x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void usmtv (const typename FieldTraits<VectorOut>::field_type& alpha,
                        const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(x,*this,y, Indices::_1,
      std::plus<>{}, [alpha](auto&& a, auto&& b) { return alpha * a * b; });
  }

  /// \brief y += alpha A^H x
  template <Concept::RandomAccessVector VectorIn,
            Concept::RandomAccessVector VectorOut>
  constexpr void usmhv (const typename FieldTraits<VectorOut>::field_type& alpha,
                        const VectorIn& x, VectorOut& y) const
      requires (extents_type::rank() == 2)
  {
    tensordotOut(x,*this,y, Indices::_1,
      std::plus<>{}, [alpha](auto&& a, auto&& b) { return alpha * a * conjugateComplex(b); });
  }

  /// @}


  /// \name tensor norms and inner products
  // @{

  /// \brief Returns the Hermitian tensor inner product with contraction over all indices `conj(A_{ij}) B_{ij}`
  template <Concept::RandomAccessTensorWithRank<extents_type::rank()> T>
    requires (Impl::checkStaticExtents<T::rank(), extents_type, typename T::extents_type>())
  constexpr Concept::Number auto inner (const T& tensor) const
  {
    auto result = tensordot(*this, tensor, std::integral_constant<std::size_t,T::rank()>{},
      std::plus<>{}, DotProduct{});
    using F = typename FieldTraits<typename decltype(result)::value_type>::field_type;
    return F(result);
  }

  /// \brief Square of 2-norm
  typename FieldTraits<value_type>::real_type two_norm2 () const
      requires (extents_type::rank() == 1)
  {
    using std::abs;
    using R = typename FieldTraits<value_type>::real_type;
    return R(abs(this->inner(*this)));
  }

  /// \brief 2-norm
  typename FieldTraits<value_type>::real_type two_norm () const
      requires (extents_type::rank() == 1)
  {
    using std::sqrt;
    return sqrt(two_norm2());
  }

  /// \brief Square of Frobenius norm
  typename FieldTraits<value_type>::real_type frobenius_norm2 () const
  {
    using std::abs;
    using R = typename FieldTraits<value_type>::real_type;
    return R(abs(this->inner(*this)));
  }

  /// \brief Frobenius norm
  typename FieldTraits<value_type>::real_type frobenius_norm () const
  {
    using std::sqrt;
    return sqrt(frobenius_norm2());
  }

  /// @}


  /// \brief Comparison of two TensorMixins for equality
  friend constexpr bool operator== (const DenseTensorMixin& lhs, const DenseTensorMixin& rhs) noexcept
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
constexpr bool operator== (const DenseTensorMixin<D,B>& lhs, const S& number) noexcept
{
  return lhs() == number;
}

// specialization for rank-0 tensor and comparison with scalar
template <Concept::Number S, class D, class B>
  requires (B::extents_type::rank() == 0)
constexpr bool operator== (const S& number, const DenseTensorMixin<D,B>& rhs) noexcept
{
  return number == rhs();
}

template <class D, class B>
struct FieldTraits< DenseTensorMixin<D,B> >
{
  using field_type = typename FieldTraits<D>::field_type;
  using real_type = typename FieldTraits<D>::real_type;
};

} // end namespace Dune

#endif // DUNE_COMMON_DENSETENSORMIXIN_HH
