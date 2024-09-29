// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TENSORDOT_HH
#define DUNE_COMMON_TENSORDOT_HH

#include <array>
#include <type_traits>

#include <dune/common/integersequence.hh>
#include <dune/common/std/extents.hh>
#include <dune/common/rangeutilities.hh>
#include <dune/common/tensor.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/concepts/tensor.hh>

namespace Dune {

namespace Impl {

/// Functors representing the multiply-add function c += a * b
/// in various forms, e.g. compound assignment, `std::fma` and with conjugation
struct MultAdd
{
  template <class A, class B>
  using result_type = std::decay_t<decltype(std::declval<A>() * std::declval<B>())>;

  template <class A, class B, class C>
  constexpr inline __attribute__((always_inline))
  void operator() (const A& a, const B& b, C& c) const noexcept
  {
    c += a * b;
  }
};

// return whether any of the static extents of E1 at positions II... or
// E2 at positions JJ... is a dynamics extent.
template <class E1, class E2, std::size_t... II, std::size_t... JJ>
constexpr bool any_dynamic (std::index_sequence<II...>, std::index_sequence<JJ...>)
{
  return (... || (E1::static_extent(II) == Std::dynamic_extent))
      || (... || (E2::static_extent(JJ) == Std::dynamic_extent));
}

// Define the result tensor type of the dot operation.
// If N indices are contracted, the first rank1-N extents of tensor1 and
// the last rank2-N extents of tensor2 are combined to build the extents
// of the output tensor. The storage type is a FixedSizeContainer if all
// extents are static, otherwise a DynamicSizeContainer.
template <class V, class E1, std::size_t... II, class E2, std::size_t... JJ>
constexpr auto makeTensor (const E1& e1, std::index_sequence<II...> iSeq, const E2& e2, std::index_sequence<JJ...> jSeq)
{
  using I = std::common_type_t<typename E1::index_type, typename E2::index_type>;
  using E = Std::extents<I, E1::static_extent(II)...,E2::static_extent(JJ)...>;
  return Tensor<V,inverseMappedExtent<E1::static_extent(II)>...,inverseMappedExtent<E2::static_extent(JJ)>...>(E{I(e1.extent(II))...,I(e2.extent(JJ))...}, V(0));
}

// Check that the dynamic extents of e1 and e2 corresponding to the positions II... and JJ... are identical
template <class E1, std::size_t... II, class E2, std::size_t... JJ>
constexpr bool checkExtents (const E1& e1, std::index_sequence<II...>, const E2& e2, std::index_sequence<JJ...>)
{
  static_assert(sizeof...(II) == sizeof...(JJ));
  using S = std::common_type_t<typename E1::index_type, typename E2::index_type>;
  return ((S(e1.extent(II)) == S(e2.extent(JJ))) && ...);
}

// Check that the static extents of E1 and E2 corresponding to the positions II... and JJ... are identical or dynamic
template <class E1, class E2, std::size_t... II, std::size_t... JJ>
constexpr bool checkStaticExtents (std::index_sequence<II...>, std::index_sequence<JJ...>)
{
  static_assert(sizeof...(II) == sizeof...(JJ));
  return ((E1::static_extent(II) == E2::static_extent(JJ) ||
           E1::static_extent(II) == Std::dynamic_extent ||
           E2::static_extent(JJ) == Std::dynamic_extent) && ...);
}


/**
 * \brief Perform a recursive nested loop over all indices:
 *
 * \b Example:
 * `C_ijk = A_ilm * B_lmjk`
 *
 * There are three sets of indices to loop over:
 * - {lm}, the indices to sum over, this corresponds to components aSeq={1,2} from A and bSeq={0,1} from B
 * - {i}, the remaining indices from A: aSeqInv={0}
 * - {jk}, the remaining indices from B: bSeqInv={2,3}
 *
 * The algorithms loops over all these indices in the corresponding ranges [0,extent]:
 * \code{.cpp}
for (int l = 0; l < a.extent(aSeq[0]); ++l)
  for (int m = 0; m < a.extent(aSeq[1]); ++m)
    for (int i = 0; i < a.extent(aSeqInv[0]); ++i)
      for (int j = 0; j < b.extent(bSeqInv[0]); ++j)
        for (int k = 0; k < b.extent(bSeqInv[1]); ++k)
          update(a(i,l,m), b(l,m,j,k), c(i,j,k))
 * \endcode
 *
 * where `update` typically means `c += a * b` but could be any ternary operation.
 **/
template <std::size_t K = 0,
          class A, class ASeq, class ASeqInv,
          class B, class BSeq, class BSeqInv,
          class C, class Updater>
constexpr inline __attribute__((always_inline))
void tensorDotImpl (const A& a, ASeq aSeq, ASeqInv aSeqInv,
                    const B& b, BSeq bSeq, BSeqInv bSeqInv,
                    C& c, const Updater& update,
                    std::array<typename A::index_type,A::rank()> aIndices = {},
                    std::array<typename B::index_type,B::rank()> bIndices = {},
                    std::array<typename C::index_type,C::rank()> cIndices = {})
{
  if constexpr(aSeq.size() > 0 && bSeq.size() > 0) {
    constexpr std::size_t I = head(aSeq);
    constexpr std::size_t J = head(bSeq);
    for (typename A::index_type k = 0; k < a.extent(I); ++k) {
      aIndices[I] = k;
      bIndices[J] = k;
      tensorDotImpl<K>(a,tail(aSeq),aSeqInv,b,tail(bSeq),bSeqInv,c,update,aIndices,bIndices,cIndices);
    }
  }
  else if constexpr(aSeqInv.size() > 0) {
    constexpr std::size_t I = head(aSeqInv);
    for (typename A::index_type i = 0; i < a.extent(I); ++i) {
      aIndices[I] = i;
      cIndices[K] = i;
      tensorDotImpl<K+1>(a,aSeq,tail(aSeqInv),b,bSeq,bSeqInv,c,update,aIndices,bIndices,cIndices);
    }
  }
  else if constexpr(bSeqInv.size() > 0) {
    constexpr std::size_t J = head(bSeqInv);
    for (typename B::index_type j = 0; j < b.extent(J); ++j) {
      bIndices[J] = j;
      cIndices[K] = j;
      tensorDotImpl<K+1>(a,aSeq,aSeqInv,b,bSeq,tail(bSeqInv),c,update,aIndices,bIndices,cIndices);
    }
  }
  else {
    update(a[aIndices], b[bIndices], c[cIndices]);
  }
}

} // end namespace Impl


/**
 * \brief Product of two tensors, stored in the output tensor.
 *
 * Product of tensors `A` and `B` with index contraction over positions `II...`
 * of `A` and positions `JJ...` of `B`. The output tensor `C` must have a rank
 * corresponding to the number of indices remaining in `A` and `B`.
 *
 * The product might be accumulated to the output tensor using the `Updater`
 * function, which by default implements an axpy operation.
 */
template <class U = Impl::MultAdd,
          Concept::Tensor A, std::size_t... II,
          Concept::Tensor B, std::size_t... JJ,
          Concept::Tensor C>
constexpr __attribute__((always_inline))
auto tensordotOut (const A& a, std::index_sequence<II...> aSeq,
                   const B& b, std::index_sequence<JJ...> bSeq,
                   C& c, U updater = {})
{
  static_assert(aSeq.size() == bSeq.size());
  Impl::tensorDotImpl(a,aSeq,difference<A::rank()>(aSeq),
                      b,bSeq,difference<B::rank()>(bSeq),
                      c,updater);
}


/**
 * \brief Product of two tensors `A` and `B` contracted over the last `N` indices
 * of `A` and the first `N` indices of `B`, stored in the output tensor.
 *
 * Product of tensors `A` and `B` with index contraction over last `N` positions
 * of `A` and first `N` positions of `B`. The output tensor `C` must have a rank
 * corresponding to the number of indices remaining in `A` and `B`.
 *
 * The product might be accumulated to the output tensor using the `Updater`
 * function, which by default implements an axpy operation.
 */
template <std::size_t N,
          class U = Impl::MultAdd,
          Concept::Tensor A, Concept::Tensor B, Concept::Tensor C>
constexpr __attribute__((always_inline))
void tensordotOut (const A& a, const B& b, C& c,
                   std::integral_constant<std::size_t,N> axes = {}, U updater = {})
{
  using SeqI = typename StaticIntegralRange<std::size_t,A::rank(),A::rank()-N>::integer_sequence;
  using InvSeqI = std::make_index_sequence<A::rank()-N>;
  using SeqJ = std::make_index_sequence<N>;
  using InvSeqJ = typename StaticIntegralRange<std::size_t,B::rank(),N>::integer_sequence;
  Impl::tensorDotImpl(a,SeqI{},InvSeqI{},b,SeqJ{},InvSeqJ{},c,updater);
}


/**
 * \brief Product of two tensors, contracted over indices II and JJ.
 *
 * Product of tensors `A` and `B` with index contraction over positions `II...`
 * of `A` and positions `JJ...` of `B`. And output tensor is constructed with
 * rank corresponding to the number of indices remaining in `A` and `B`.
 *
 * The product might be accumulated to the output tensor using the `Updater`
 * function, which by default implements an axpy operation.
 *
 * \b Examples:
 * - outer product of 2-tensors: `c(i,j,k,l) = a(i,j) * b(k,l)`
 *     is `tensordot(a,index_sequence<>{},b,index_sequence<>{})`
 * - matrix-matrix product: `c(i,j) = a(i,k) * b(k,j)`
 *     is `tensordot(a,index_sequence<1>{},b,index_sequence<0>{})`
 * - inner product of 2-tensors: `c = a(i,j) * b(i,j)`
 *     is `tensordot(a,index_sequence<0,1>{},b,index_sequence<0,1>{})`
 */
template <class U = Impl::MultAdd,
          Concept::Tensor A, std::size_t... II,
          Concept::Tensor B, std::size_t... JJ>
constexpr auto tensordot (const A& a, std::index_sequence<II...> aSeq,
                          const B& b, std::index_sequence<JJ...> bSeq,
                          U updater = {})
{
  // the last extents(II) of a and the extents(JJ) of b must match
  static_assert(Impl::checkStaticExtents<typename A::ExtentsType,typename B::ExtentsType>(aSeq, bSeq));
  assert((Impl::checkExtents(a.extents(), aSeq, b.extents(), bSeq)));

  // create result tensor by collecting the extents of a and b that are not folded
  using V = typename U::template result_type<typename A::ElementType,typename B::ElementType>;
  auto c = Impl::makeTensor<V>(a.extents(), difference<A::rank()>(aSeq),
                               b.extents(), difference<B::rank()>(bSeq));
  Impl::tensorDotImpl(a,aSeq,difference<A::rank()>(aSeq),b,bSeq,difference<B::rank()>(bSeq),c,updater);
  return c;
}


/**
 *  \brief Product of two tensors `A` and `B` contracted over the last `N` indices
 * of `A` and the first `N` indices of `B`
 *
 * Sum over the last `N` axes of `a` and the first `N` axes of `b`
 *
 * \b Examples:
 * - outer product of 2-tensors: `c(i,j,k,l) = a(i,j) * b(k,l)` is `tensordot<0>(a,b)`
 * - matrix-matrix product: `c(i,j) = a(i,k) * b(k,j)` is `tensordot<1>(a,b)`
 * - inner product of 2-tensors: `c = a(i,j) * b(i,j)` is `tensordot<2>(a,b)`
 **/
template <std::size_t N,
          class U = Impl::MultAdd,
          Concept::Tensor A, Concept::Tensor B>
constexpr auto tensordot (const A& a, const B& b,
                          std::integral_constant<std::size_t,N> axes = {},
                          U updater = {})
{
  using SeqI = typename StaticIntegralRange<std::size_t,A::rank(),A::rank()-N>::integer_sequence;
  using SeqJ = std::make_index_sequence<N>;
  return tensordot(a,SeqI{},b,SeqJ{},updater);
}

} // end namespace Dune

#endif // DUNE_COMMON_TENSORDOT_HH
