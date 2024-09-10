// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_INITIALIZER_LIST_HH
#define DUNE_COMMON_INITIALIZER_LIST_HH

#include <cassert>
#include <initializer_list>

namespace Dune {
namespace Impl {

template <class Value, int rank>
struct NestedInitializerList
{
  using type = std::initializer_list<typename NestedInitializerList<Value,rank-1>::type>;
};

template <class Value>
struct NestedInitializerList<Value,1>
{
  using type = std::initializer_list<Value>;
};

template <class Value>
struct NestedInitializerList<Value,0>
{
  using type = Value;
};

} // end namespace Impl

/// \brief A nested `std::initializer_list<std::initializer_list<...>>` up to depth `rank`
template <class Value, int rank>
using NestedInitializerList_t = typename Impl::NestedInitializerList<Value,rank>::type;


/// \brief A utility to recursively unpack nested initializer lists
template <class Value, class Extents, int I = Extents::rank()>
class InitializerList
{
public:
  using value_type = Value;
  using extents_type = Extents;

  template <class F>
  static void apply (NestedInitializerList_t<Value,I> values, const extents_type& extents, F&& set_value)
  {
    assert(values.size() == std::size_t(extents.extent(extents_type::rank()-I)));

    // process all the sub lists
    for (auto&& sub : values)
      InitializerList<value_type,extents_type,I-1>::apply(sub, extents, set_value);
  }
};

#ifndef DOXYGEN
template <class Value, class Extents>
class InitializerList<Value,Extents,1>
{
public:
  using value_type = Value;
  using extents_type = Extents;

  template <class F>
  static void apply (std::initializer_list<Value> values, const extents_type& extents, F&& set_value)
  {
    assert(values.size() == std::size_t(extents.extent(extents_type::rank()-1)));

    for (auto&& value : values)
      set_value(value);
  }
};

template <class Value, class Extents>
class InitializerList<Value,Extents,0>
{
public:
  using value_type = Value;
  using extents_type = Extents;

  template <class F>
  static void apply (const Value& value, const extents_type& extents, F&& set_value)
  {
    set_value(value);
  }
};
#endif // DOXYGEN

} // end namespace Dune

#endif // DUNE_COMMON_INITIALIZER_LIST_HH
