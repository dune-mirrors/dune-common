// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_CONCEPTS_ARCHETYPES_MULTIDIMENSIONAL_HH
#define DUNE_COMMON_CONCEPTS_ARCHETYPES_MULTIDIMENSIONAL_HH

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include <dune/common/std/layout_right.hh>

namespace Dune::Concept::Archetypes {

/**
 * \brief A minimal model of the concept \ref Dune::Concept::MultiDimensional.
 * \ingroup CxxConcepts
 *
 * The archetype is a non-owning view of a contiguous array of elements in row-major order.
 * It provides only the interface required by the concept. Instantiating generic code with
 * this archetype checks that the code does not use any methods beyond that interface,
 * e.g., `size()`, `begin()`, `end()` or chained `operator[]`.
 *
 * \tparam Element  The element type, possibly const-qualified.
 * \tparam Extents  The index space, a specialization of `Std::extents`.
 *
 * \b Example:
 * \code{.cpp}
   std::array<double,6> data{};
   Archetypes::MultiDimensional<double, Std::extents<std::size_t,2,3>> m{data.data()};
   m[std::array{1,2}] = 1.0;
 * \endcode
 **/
template <class Element, class Extents>
class MultiDimensional
{
  using mapping_type = typename Std::layout_right::template mapping<Extents>;

public:
  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using rank_type = typename extents_type::rank_type;
  using value_type = std::remove_cv_t<Element>;

  //! Construct the archetype from the data pointer and the extents
  constexpr explicit MultiDimensional (Element* data, const extents_type& e = extents_type{})
    : data_{data}
    , mapping_{e}
  {}

  static constexpr rank_type rank () noexcept { return extents_type::rank(); }
  static constexpr rank_type rank_dynamic () noexcept { return extents_type::rank_dynamic(); }
  static constexpr std::size_t static_extent (rank_type r) noexcept { return extents_type::static_extent(r); }

  constexpr index_type extent (rank_type r) const noexcept { return extents().extent(r); }
  constexpr const extents_type& extents () const noexcept { return mapping_.extents(); }

  //! Access element at position `{i0,i1,...}`
  template <class Index>
    requires std::is_convertible_v<const Index&, index_type>
  constexpr Element& operator[] (const std::array<Index,extents_type::rank()>& indices) const
  {
    return std::apply([&](auto... ii) -> Element& {
      return data_[mapping_(index_type(ii)...)]; }, indices);
  }

private:
  Element* data_;
  mapping_type mapping_;
};

} // end namespace Dune::Concept::Archetypes

#endif // DUNE_COMMON_CONCEPTS_ARCHETYPES_MULTIDIMENSIONAL_HH
