// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_CONCEPTS_ARCHETYPES_TENSOR_HH
#define DUNE_COMMON_CONCEPTS_ARCHETYPES_TENSOR_HH

#include <array>

namespace Dune::Concept::Archetypes {

template <std::size_t r>
struct Extents
{
  using rank_type = std::size_t;
  using index_type = std::size_t;

  static constexpr std::size_t rank () { return r; }
  static constexpr std::size_t static_extent (std::size_t) { return 0; }
  index_type extent (std::size_t) const;
};

template <class V, std::size_t r>
struct Tensor : Archetypes::Extents<r>
{
  using reference = V&;
  using const_reference = const V&;

  using extents_type = Archetypes::Extents<r>;
  using index_type = typename extents_type::index_type;

  const extents_type& extents ();

  reference operator[] (std::array<index_type,r>);
  const_reference operator[] (std::array<index_type,r>) const;
};


} // end namespace Dune::Concept::Archetypes

#endif // DUNE_COMMON_CONCEPTS_ARCHETYPES_TENSOR_HH
