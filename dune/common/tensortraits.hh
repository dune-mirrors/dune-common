// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_TENSORTRAITS_HH
#define DUNE_COMMON_TENSORTRAITS_HH

#include <cstddef>

namespace Dune {

template <class T>
struct TensorTraits
{
  using extents_type = typename T::extents_type;
  using index_type = typename extents_type::index_type;
  using rank_type = typename extents_type::rank_type;


  /// \brief Number of elements in all dimensions of the array, \related extents
  static constexpr const extents_type& extents (const T& tensor) noexcept { return tensor.extents(); }

  /// \brief Number of dimensions of the array
  static constexpr rank_type rank () noexcept { return extents_type::rank(); }

  /// \brief Number of dimension with dynamic size
  static constexpr rank_type rank_dynamic () noexcept { return extents_type::rank_dynamic(); }

  /// \brief Number of elements in the r'th dimension of the tensor
  static constexpr std::size_t static_extent (rank_type r) noexcept { return extents_type::static_extent(r); }

  /// \brief Number of elements in the r'th dimension of the tensor
  static constexpr index_type extent (const T& tensor, rank_type r) noexcept { return extents(tensor).extent(r); }
};

} // end namespace Dune

#endif // DUNE_COMMON_TENSORTRAITS_HH
