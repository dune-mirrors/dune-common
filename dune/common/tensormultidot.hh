// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TENSORMULTIDOT_HH
#define DUNE_COMMON_TENSORMULTIDOT_HH

#include <span>
#include <type_traits>
#include <utility>

#include <dune/common/promotiontraits.hh>
#include <dune/common/tensordot.hh>
#include <dune/common/tensortraits.hh>

namespace Dune {

template <class Element, std::size_t... extents>
class DenseTensor;

namespace Impl {

struct TensorMultiDot
{
  template <class A, class T0>
  static auto applyImpl (const A& a, const T0& t0)
  {
    return tensordot(a, std::index_sequence<0>{}, t0, std::index_sequence<0>{});
  }

  template <class A, class T0, class... Ts>
  static auto applyImpl (const A& a, const T0& t0, const Ts&... ts)
  {
    return applyImpl(applyImpl(a, t0), ts...);
  }

  // recursive application of tensordot
  template <class A, class... Ts>
  static auto apply (const A& a, const Ts&... ts)
  {
    return applyImpl(a,ts...);
  }

  // specialization for `Vector^T * Matrix * Vector`
  template <Concept::Matrix A, Concept::VectorLike T0, Concept::VectorLike T1>
  static auto apply (const A& a, const T0& t0, const T1& t1)
  {
    static_assert(A::static_extent(0) == std::dynamic_extent || TensorTraits<T0>::static_extent(0) == std::dynamic_extent || A::static_extent(0) == TensorTraits<T0>::static_extent(0));
    static_assert(A::static_extent(1) == std::dynamic_extent || TensorTraits<T1>::static_extent(0) == std::dynamic_extent || A::static_extent(1) == TensorTraits<T1>::static_extent(0));

    using E0 = typename Dune::PromotionTraits<typename A::element_type, std::decay_t<decltype(t0[0])>>::PromotedType;
    using E1 = typename Dune::PromotionTraits<typename A::element_type, std::decay_t<decltype(t1[0])>>::PromotedType;
    using Element = typename Dune::PromotionTraits<E0,E1>::PromotedType;

    Dune::DenseTensor<Element> out{};

    for (typename A::index_type i = 0; i < a.extent(0); ++i)
      for (typename A::index_type j = 0; j < a.extent(1); ++j)
        out() += a(i,j) * t0[i] * t1[j];

    return out;
  }

  // specialization for `Matrix^T * Matrix * Matrix`
  template <Concept::Matrix A, Concept::MatrixLike T0, Concept::MatrixLike T1>
  static auto apply (const A& a, const T0& t0, const T1& t1)
  {
    static_assert(A::static_extent(0) == std::dynamic_extent || TensorTraits<T0>::static_extent(0) == std::dynamic_extent || A::static_extent(0) == TensorTraits<T0>::static_extent(0));
    static_assert(A::static_extent(1) == std::dynamic_extent || TensorTraits<T1>::static_extent(0) == std::dynamic_extent || A::static_extent(1) == TensorTraits<T1>::static_extent(0));

    using E0 = typename Dune::PromotionTraits<typename A::element_type, std::decay_t<decltype(t0[std::array{0,0}])>>::PromotedType;
    using E1 = typename Dune::PromotionTraits<typename A::element_type, std::decay_t<decltype(t1[std::array{0,0}])>>::PromotedType;
    using Element = typename Dune::PromotionTraits<E0,E1>::PromotedType;

    Dune::DenseTensor<Element, TensorTraits<T0>::static_extent(1), TensorTraits<T1>::static_extent(1)> out{TensorTraits<T1>::extent(t0,0),TensorTraits<T1>::extent(t1,0)};

    for (typename A::index_type i = 0; i < a.extent(0); ++i)
      for (typename A::index_type j = 0; j < a.extent(1); ++j)
        for (typename A::index_type k = 0; k < TensorTraits<T1>::extent(t0,1); ++k)
          for (typename A::index_type l = 0; j < TensorTraits<T1>::extent(t1,1); ++l)
            out(k,l) += a(i,j) * t0[std::array{i,k}] * t1[std::array{j,l}];

    return out;
  }

};

} // end namespace Impl

template <Concept::Tensor A, Concept::TensorLike... Ts>
  requires (sizeof...(Ts) == A::rank())
Concept::Tensor auto tensorMultiDot (const A& a, const Ts&... ts)
{
  if constexpr (sizeof...(Ts) == 0)
    return a;
  else
    return Impl::TensorMultiDot::apply(a, ts...);
}

} // end namespace Dune

#endif // DUNE_COMMON_TENSORMULTIDOT_HH
