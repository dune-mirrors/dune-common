// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TENSOR_HH
#define DUNE_COMMON_TENSOR_HH

#include <array>
#include <concepts>
#include <span>
#include <type_traits>
#include <vector>

#include <dune/common/initializerlist.hh>
#include <dune/common/tensormixin.hh>
#include <dune/common/tensorspan.hh>
#include <dune/common/concepts/number.hh>
#include <dune/common/std/extents.hh>
#include <dune/common/std/layout_right.hh>
#include <dune/common/std/mdarray.hh>

namespace Dune {
namespace Impl {

template <class Element, std::size_t... extents>
struct TensorStorageType
{
  using container_type = std::conditional_t<((extents == std::dynamic_extent) || ...),
    std::vector<Element>, std::array<Element, (extents * ... * std::size_t(1))>>;
  using extents_type = Std::extents<typename container_type::size_type,extents...>;
  using type = Std::mdarray<Element, extents_type, Std::layout_right, container_type>;
};

} // end namespace Impl


/**
 * \brief A tensor of arbitrary rank and individual dimension extents, static
 * or dynamic, storing field values in a container of corresponding static or
 * dynamic size.
 *
 * \ingroup Tensors
 * \nosubgrouping
 *
 * \tparam Element  The element type stored in the tensor
 * \tparam extents  Individual static extents or std::dynamic_extent
 **/
template <class Element, std::size_t... extents>
class Tensor
    : public TensorMixin<Tensor<Element,extents...>,
        typename Impl::TensorStorageType<Element,extents...>::type>
{
  using self_type = Tensor;
  using storage_type = typename Impl::TensorStorageType<Element,extents...>::type;
  using base_type = TensorMixin<self_type, storage_type>;

public:
  using element_type = typename base_type::element_type;
  using extents_type = typename base_type::extents_type;
  using value_type = typename base_type::value_type;
  using index_type = typename base_type::index_type;
  using layout_type = typename base_type::layout_type;
  using mapping_type = typename base_type::mapping_type;

  using tensorspan_type = TensorSpan<element_type,extents_type,layout_type>;
  using const_tensorspan_type = TensorSpan<const element_type,extents_type,layout_type>;

public:
  /// \brief Inherit constructor from TensorMixin
  using base_type::base_type;

  /// \name Additional Tensor constructors
  /// @{

  explicit constexpr Tensor (const value_type& value)
    : base_type{extents_type{}, value}
  {}

  /**
   * \brief Constructor from a brace-init list of values
   *
   * \b Example:
   * \code{.cpp}
   Tensor<double,2,2> matrix{
     {1.0,2.0},
     {2.0, 3.0}
   };
   * \endcode
   **/
  template <class E = extents_type>
    requires (E::rank_dynamic() == 0 && std::is_default_constructible_v<E>)
  constexpr Tensor (NestedInitializerList_t<value_type,extents_type::rank()> init)
    : Tensor{extents_type{}, init}
  {}

  /**
   * \brief Constructor from an extents and brace-init list of values
   *
   * \b Example:
   * \code{.cpp}
   using Extents = typename Tensor<double,dynamic_extent,dynamic_extent>::extents_type;
   Tensor<double,dynamic_extent,dynamic_extent> matrix(Extents{2,2},
   {
     {1.0,2.0},
     {2.0, 3.0}
   });
   * \endcode
   **/
  template <class M = mapping_type>
    requires (std::is_constructible_v<M,extents_type>)
  constexpr Tensor (const extents_type& e, NestedInitializerList_t<value_type,extents_type::rank()> init)
    : Tensor{mapping_type{e}, init}
  {}

  /**
   * \brief Constructor from a span of extents and brace-init list of values
   *
   * \b Example:
   * \code{.cpp}
   Tensor<double,dynamic_extent,dynamic_extent> matrix(std::array{2,2},
   {
     {1.0,2.0},
     {2.0, 3.0}
   });
   * \endcode
   **/
  template <std::convertible_to<index_type> Otherindex_type, std::size_t N>
    requires (std::is_nothrow_constructible_v<index_type,const Otherindex_type&> &&
              (N == extents_type::rank_dynamic() || N == extents_type::rank()))
  constexpr Tensor (std::span<Otherindex_type,N> e,
                    NestedInitializerList_t<value_type,extents_type::rank()> init)
    : Tensor{mapping_type{extents_type{e}}, init}
  {}

  /**
   * \brief Constructor from a layout-mapping and brace-init list of values
   *
   * \b Example:
   * \code{.cpp}
   using Extents = typename Tensor<double,dynamic_extent,dynamic_extent>::extents_type;
   using Mapping = typename Tensor<double,dynamic_extent,dynamic_extent>::mapping_type;
   Tensor<double,dynamic_extent,dynamic_extent> matrix(Mapping{Extents{2,2}},
   {
     {1.0,2.0},
     {2.0, 3.0}
   });
   * \endcode
   **/
  constexpr Tensor (const mapping_type& m, NestedInitializerList_t<value_type,extents_type::rank()> init)
    : base_type{m}
  {
    auto it = this->container_data();
    InitializerList<value_type,extents_type>::apply(init,this->extents(),
      [&it](value_type value) { *it++ = value; });
  }

  /// \brief Converting constructor from another Tensor
  template <class OtherElement, std::size_t... otherExtents>
  constexpr Tensor (const Tensor<OtherElement,otherExtents...>& other)
    : base_type{other.toTensorSpan()}
  {}

  /// \brief Copy constructor with default behavior
  constexpr Tensor (const Tensor&) = default;

  /// \brief Move constructor with default behavior
  constexpr Tensor (Tensor&&) = default;

  /// @}


  /// \name Assignment operators
  /// @{

  /// \brief Assignment operators inherited from the mixin class and storage class
  using base_type::operator=;

  /// \brief Copy assignment-operator with default behavior
  constexpr Tensor& operator= (const Tensor&) = default;

  /// \brief Move assignment-operator with default behavior
  constexpr Tensor& operator= (Tensor&&) = default;

  /// @}


  /// \name Multi index access
  /// @{

  /**
   * \brief Subscript operator to access the tensor components using a single index or an array of indices.
   * \b Examples:
   * \code{c++}
     Tensor<double,3,3> matrix;
     matrix[std::array{0,1}] = 7.0;

     Tensor<double,3> vector;
     vector[2] = 42.0;
     \endcode
   **/
  using base_type::operator[];

  /**
   * \brief Access an element of the tensor using a variadic list of indices.
   * \b Examples:
   * \code{c++}
     Tensor<double,3,3,3> tensor;
     tensor(0,1,2) = 42.0;
     \endcode
   **/
  using base_type::operator();

  /// @}


  /// \name Modifiers
  /// @{

  /**
    * \brief Change the extents of the tensor and resize the underlying container with given default value.
    *
    * \note This does not allow to resize tensors with purely static extents, since
    * the new extents type must be identical to the old extents type and all dimensions
    * are encoded in that type. However, a `.resize()` is still called on the underlying
    * container with the extra `value` argument.
    **/
  template <std::convertible_to<value_type> V>
  void resize (const extents_type& e, const V& value)
  {
    auto container = std::move(*this).extract_container();
    auto m = mapping_type{e};
    container.resize(m.required_span_size(), value);
    static_cast<base_type&>(*this) = base_type{m, std::move(container)};
  }

  /**
    * \brief Change the extents of the tensor and resize the underlying container.
    *
    * This resizes the underlying container with the new entries initialized to 0.
    *
    * \note This does not allow to resize tensors with purely static extents, since
    * the new extents type must be identical to the old extents type and all dimensions
    * are encoded in that type.
    **/
  void resize (const extents_type& e)
  {
    resize(e, value_type(0));
  }

  /**
    * \brief Change the extents of the tensor by the given individual extents.
    *
    * \note This does not allow to resize tensors with purely static extents, since
    * the new extents type must be identical to the old extents type and all dimensions
    * are encoded in that type. However, a `.resize()` is still called on the underlying
    * container with the extra `value` argument.
    **/
  template <std::convertible_to<index_type>... index_types, std::convertible_to<value_type> V>
    requires ((sizeof...(index_types) == extents_type::rank() ||
               sizeof...(index_types) == extents_type::rank_dynamic()) &&
              std::is_constructible_v<extents_type,index_types...>)
  void resize (index_types... exts, const V& value)
  {
    resize(extents_type{exts...}, value);
  }

  /**
    * \brief Change the extents of the tensor by the given individual extents.
    *
    * This resizes the underlying container with the new entries initialized to 0.
    *
    * \note This does not allow to resize tensors with purely static extents, since
    * the new extents type must be identical to the old extents type and all dimensions
    * are encoded in that type.
    **/
  template <std::convertible_to<index_type>... index_types>
    requires ((sizeof...(index_types) == extents_type::rank() ||
               sizeof...(index_types) == extents_type::rank_dynamic()) &&
              std::is_constructible_v<extents_type,index_types...>)
  void resize (index_types... exts)
  {
    resize(extents_type{exts...}, value_type(0));
  }

  /// @}


  /// \name Conversion into mdspan
  /// @{

  /// \brief Conversion operator to TensorSpan
  template <class V, class E, class L, class A>
    requires std::constructible_from<TensorSpan<V,E,L,A>, tensorspan_type>
  constexpr operator TensorSpan<V,E,L,A> ()
  {
    return tensorspan_type(this->container_data(), this->mapping());
  }

  /// \brief Conversion operator to TensorSpan
  template <class V, class E, class L, class A>
    requires std::constructible_from<TensorSpan<V,E,L,A>, const_tensorspan_type>
  constexpr operator TensorSpan<V,E,L,A> () const
  {
    return const_tensorspan_type(this->container_data(), this->mapping());
  }

  /// \brief Conversion function to TensorSpan
  template <class A = Std::default_accessor<element_type>>
    requires std::assignable_from<tensorspan_type&, TensorSpan<element_type,extents_type,layout_type,A>>
  constexpr TensorSpan<element_type,extents_type,layout_type,A>
  toTensorSpan (const A& accessor = A{})
  {
    return TensorSpan<element_type,extents_type,layout_type,A>(this->container_data(), this->mapping(), accessor);
  }

  /// \brief Conversion function to TensorSpan
  template <class A = Std::default_accessor<const element_type>>
    requires std::assignable_from<const_tensorspan_type&, TensorSpan<const element_type,extents_type,layout_type,A>>
  constexpr TensorSpan<const element_type,extents_type,layout_type,A>
  toTensorSpan (const A& accessor = A{}) const
  {
    return TensorSpan<const element_type,extents_type,layout_type,A>(this->container_data(), this->mapping(), accessor);
  }

  /// @}
};


namespace Concept {

template <class M>
concept LayoutMapping = std::same_as<M,
      typename M::layout_type::template mapping<typename M::extents_type>>;

} // end namespace Concept


/// \name Deduction guides
/// \relates Tensor
/// @{

template <class index_type, std::size_t... exts, class value_type>
Tensor (Std::extents<index_type, exts...>, value_type)
  -> Tensor<value_type, exts...>;

// NOTE: since deduction guides cannot be unrolled or defined inside a
// helper class, we need to write out the specialization for the different
// ranks. This is done up to tensor rank 3. Higher order ranks cannot rely
// on CTAD.

template <Concept::LayoutMapping Mapping, Concept::Number value_type>
  requires (Mapping::extents_type::rank() == 0)
Tensor (Mapping, value_type)
  -> Tensor<value_type>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type>
  requires (Mapping::extents_type::rank() == 1)
Tensor (Mapping, value_type)
  -> Tensor<value_type, Mapping::extents_type::static_extent(0)>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type>
  requires (Mapping::extents_type::rank() == 2)
Tensor (Mapping, value_type)
  -> Tensor<value_type,
      Mapping::extents_type::static_extent(0),
      Mapping::extents_type::static_extent(1)>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type>
  requires (Mapping::extents_type::rank() == 3)
Tensor (Mapping, value_type)
  -> Tensor<value_type,
      Mapping::extents_type::static_extent(0),
      Mapping::extents_type::static_extent(1),
      Mapping::extents_type::static_extent(2)>;

template <class index_type, std::size_t... exts, Concept::Number value_type, class Alloc>
Tensor (Std::extents<index_type, exts...>, value_type, const Alloc&)
  -> Tensor<value_type, exts...>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type, class Alloc>
  requires (Mapping::extents_type::rank() == 0)
Tensor (Mapping, value_type, const Alloc&)
  -> Tensor<value_type>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type, class Alloc>
  requires (Mapping::extents_type::rank() == 1)
Tensor (Mapping, value_type, const Alloc&)
  -> Tensor<value_type, Mapping::extents_type::static_extent(0)>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type, class Alloc>
  requires (Mapping::extents_type::rank() == 2)
Tensor (Mapping, value_type, const Alloc&)
  -> Tensor<value_type,
      Mapping::extents_type::static_extent(0),
      Mapping::extents_type::static_extent(1)>;

template <Concept::LayoutMapping Mapping, Concept::Number value_type, class Alloc>
  requires (Mapping::extents_type::rank() == 3)
Tensor (Mapping, value_type, const Alloc&)
  -> Tensor<value_type,
      Mapping::extents_type::static_extent(0),
      Mapping::extents_type::static_extent(1),
      Mapping::extents_type::static_extent(2)>;

template <class V, class I, std::size_t... exts, class L, class C>
Tensor (Std::mdarray<V,Dune::Std::extents<I,exts...>,L,C>)
  -> Tensor<V, exts...>;

template <class V, class I, std::size_t... exts, class L, class C>
Tensor (Std::mdspan<V,Dune::Std::extents<I,exts...>,L,C>)
  -> Tensor<std::remove_cv_t<V>, exts...>;

/// @}


template <class V, std::size_t... exts>
TensorSpan (Tensor<V,exts...>&)
  -> TensorSpan<V,typename Impl::TensorStorageType<V,exts...>::extents_type,Std::layout_right,Std::default_accessor<V>>;

template <class V, std::size_t... exts>
TensorSpan (const Tensor<V,exts...>&)
  -> TensorSpan<const V,typename Impl::TensorStorageType<V,exts...>::extents_type,Std::layout_right,Std::default_accessor<const V>>;


template <class V, std::size_t... exts>
struct FieldTraits< Tensor<V,exts...> >
{
  using field_type = typename FieldTraits<V>::field_type;
  using real_type = typename FieldTraits<V>::real_type;
};

} // end namespace Dune

#endif // DUNE_COMMON_TENSOR_HH
