// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TEST_FOREACHINDEX_HH
#define DUNE_COMMON_TEST_FOREACHINDEX_HH

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#include <dune/common/std/extents.hh>
#include <dune/common/std/span.hh>
#include <dune/common/std/functional.hh>

namespace Dune {
namespace Impl {

template <class Extents, class Fun, class... Indices>
inline __attribute__((always_inline))
void forEachIndexImpl (const Extents& extents, Fun f, Indices... ii)
{
  constexpr typename Extents::rank_type pos = sizeof...(Indices);
  if constexpr(pos < Extents::rank()) {
    if constexpr(Extents::static_extent(pos) == Std::dynamic_extent)
      for (typename Extents::index_type i = 0; i < extents.extent(pos); ++i)
        forEachIndexImpl(extents, std::move(f), ii...,i);
    else
      for (std::size_t i = 0; i < Extents::static_extent(pos); ++i)
        forEachIndexImpl(extents, std::move(f), ii...,typename Extents::index_type(i));
  }
  else {
    using I = std::array<typename Extents::index_type,Extents::rank()>;
    std::invoke(f, I{ii...});
  }
}

} // end namespace Impl


/// \brief Invoke the function `f` on all index-tuples in the multi dimensional index-space given by `extents`.
template <class Extents, class Fun,
  class I = std::array<typename Extents::index_type,Extents::rank()>,
  std::enable_if_t<std::is_invocable_v<Fun,I>, int> = 0>
inline __attribute__((always_inline))
void forEachIndex (const Extents& extents, Fun f)
{
  Impl::forEachIndexImpl(extents, std::move(f));
}

} // end namespace Dune

#endif // DUNE_COMMON_TEST_FOREACHINDEX_HH
