// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_COMMON_MULTIDIMENSIONAL_HH
#define DUNE_COMMON_MULTIDIMENSIONAL_HH

#include <array>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <dune/common/iteratorfacades.hh>
#include <dune/common/rangeutilities.hh>
#include <dune/common/concepts/multidimensional.hh>
#include <dune/common/std/extents.hh>
#include <dune/common/std/mdspan.hh>

/**
 * \file
 * \brief Utilities for objects modeling the concept `Concept::MultiDimensional`.
 *
 * The functions in this file accept all models of `Concept::MultiDimensional`, e.g.,
 * `Std::mdspan`, `Std::mdarray`, `FieldVector` or `FieldMatrix`, and additionally
 * contiguous one-dimensional containers like `std::vector` and `std::array`, that are
 * adapted by `asMultiDimensional()`.
 */

namespace Dune {

namespace Impl {

  // The static size of a contiguous range, or std::dynamic_extent
  template <class R>
  struct ContiguousRangeStaticSize
    : std::integral_constant<std::size_t, std::dynamic_extent> {};

  template <class R>
    requires requires { std::tuple_size<R>::value; }
  struct ContiguousRangeStaticSize<R>
    : std::integral_constant<std::size_t, std::tuple_size<R>::value> {};

  template <class T, std::size_t N>
  struct ContiguousRangeStaticSize<std::span<T,N>>
    : std::integral_constant<std::size_t, N> {};

  // A contiguous range that is not already a model of MultiDimensional
  template <class R>
  concept ContiguousRange = !Concept::MultiDimensional<R> &&
  requires(R& r)
  {
    { std::data(r) } -> std::convertible_to<const volatile void*>;
    { std::size(r) } -> std::convertible_to<std::size_t>;
  };

} // end namespace Impl


/**
 * \brief Return an object modeling `Concept::MultiDimensional` representing `t`.
 *
 * Models of `Concept::MultiDimensional` are returned as they are, by reference for
 * lvalue arguments and moved into a new object for rvalue arguments.
 */
template <class T>
  requires Concept::MultiDimensional<std::remove_cvref_t<T>>
constexpr T& asMultiDimensional (T& t) noexcept
{
  return t;
}

//! \copydoc asMultiDimensional(T&)
template <class T>
  requires (Concept::MultiDimensional<T> && !std::is_lvalue_reference_v<T>)
constexpr T asMultiDimensional (T&& t) noexcept(std::is_nothrow_move_constructible_v<T>)
{
  return T(std::move(t));
}

/**
 * \brief Return a one-dimensional `Std::mdspan` viewing the elements of a contiguous range.
 *
 * Contiguous ranges, e.g. `std::vector`, `std::array`, or `std::span`, are represented by
 * a rank-1 `Std::mdspan` with static extent if the size is known at compile time. Since
 * the returned object refers to the elements of the range, the range must be an lvalue
 * or a borrowed range, like `std::span`.
 */
template <class R>
  requires (Impl::ContiguousRange<std::remove_cvref_t<R>> &&
    (std::is_lvalue_reference_v<R> || std::ranges::enable_borrowed_range<std::remove_cvref_t<R>>))
constexpr auto asMultiDimensional (R&& r)
{
  using Element = std::remove_pointer_t<decltype(std::data(r))>;
  constexpr std::size_t n = Impl::ContiguousRangeStaticSize<std::remove_cvref_t<R>>::value;
  if constexpr(n == std::dynamic_extent)
    return Std::mdspan<Element, Std::dims<1>>(std::data(r), std::size(r));
  else
    return Std::mdspan<Element, Std::extents<std::size_t,n>>(std::data(r));
}


namespace Concept {

/**
 * \brief Types that can be represented as `MultiDimensional` by `asMultiDimensional()`.
 * \ingroup CxxConcepts
 *
 * The concept is satisfied if an lvalue of type `T` can be passed to `asMultiDimensional()`.
 * In addition to models of \ref MultiDimensional, this includes contiguous ranges like
 * `std::vector` and `std::array`.
 *
 * \note In contrast to \ref MultiDimensional, the type `T` might be a reference or cv-qualified type.
 *
 * \experimental
 * \hideinitializer
 **/
template <class T>
concept MultiDimensionalLike = requires(std::remove_reference_t<T>& t)
{
  { Dune::asMultiDimensional(t) };
};

} // end namespace Concept


/**
 * \brief The number of elements in the index space of `t`, i.e., the product of all extents.
 *
 * This is equivalent to `mdspan::size()`. Note that the method `size()` of some models of
 * `Concept::MultiDimensional` has a different meaning, e.g., `FieldMatrix::size()` returns
 * the number of rows.
 */
template <Concept::MultiDimensionalLike T>
constexpr std::size_t indexSpaceSize (const T& t)
{
  const auto& md = asMultiDimensional(t);
  using MD = std::remove_cvref_t<decltype(md)>;
  std::size_t s = 1;
  for (typename MD::rank_type r = 0; r < MD::rank(); ++r)
    s *= std::size_t(md.extent(r));
  return s;
}


namespace Impl {

  // Iterator over all multi-indices of an index space in row-major order
  template <class Extents>
  class IndexSpaceIterator
    : public Dune::IteratorFacade<IndexSpaceIterator<Extents>, std::forward_iterator_tag,
        std::array<typename Extents::index_type, Extents::rank()>,
        const std::array<typename Extents::index_type, Extents::rank()>&>
  {
    using index_type = typename Extents::index_type;
    using multi_index_type = std::array<index_type, Extents::rank()>;

  public:
    constexpr IndexSpaceIterator () = default;

    constexpr IndexSpaceIterator (const Extents& extents, std::size_t position)
      : extents_{extents}
      , position_{position}
      , index_{}
    {}

    constexpr const multi_index_type& operator* () const noexcept { return index_; }

    constexpr IndexSpaceIterator& operator++ () noexcept
    {
      ++position_;
      for (std::size_t r = Extents::rank(); r > 0; --r) {
        if (++index_[r-1] < extents_.extent(r-1))
          break;
        if (r > 1)
          index_[r-1] = 0;
      }
      return *this;
    }

    constexpr bool operator== (const IndexSpaceIterator& other) const noexcept
    {
      return position_ == other.position_;
    }

  private:
    Extents extents_ = {};
    std::size_t position_ = 0;
    multi_index_type index_ = {};
  };

  // Range over all multi-indices of an index space
  template <class Extents>
  class IndexSpaceRange
  {
  public:
    using iterator = IndexSpaceIterator<Extents>;
    using const_iterator = iterator;

    constexpr explicit IndexSpaceRange (const Extents& extents)
      : extents_{extents}
    {}

    constexpr iterator begin () const noexcept { return iterator{extents_, 0}; }
    constexpr iterator end () const noexcept { return iterator{extents_, size()}; }

    constexpr std::size_t size () const noexcept
    {
      std::size_t s = 1;
      for (std::size_t r = 0; r < Extents::rank(); ++r)
        s *= std::size_t(extents_.extent(r));
      return s;
    }

  private:
    Extents extents_;
  };

  // Store the result of `asMultiDimensional()`, references as pointer and values by value
  template <class T>
  class MultiDimensionalStorage
  {
  public:
    constexpr explicit MultiDimensionalStorage (T&& t) : value_{std::move(t)} {}
    constexpr const T& get () const noexcept { return value_; }
  private:
    T value_;
  };

  template <class T>
  class MultiDimensionalStorage<T&>
  {
  public:
    constexpr explicit MultiDimensionalStorage (T& t) : ptr_{&t} {}
    constexpr T& get () const noexcept { return *ptr_; }
  private:
    T* ptr_;
  };

  template <class T>
  constexpr auto storeMultiDimensional (T& t)
  {
    using Result = decltype(asMultiDimensional(t));
    return MultiDimensionalStorage<Result>{asMultiDimensional(t)};
  }

  // A view of the row `i` of a rank-2 multi-dimensional object
  template <class Storage>
  class RowView
  {
    using matrix_type = std::remove_cvref_t<decltype(std::declval<const Storage&>().get())>;

  public:
    using index_type = typename matrix_type::index_type;
    using rank_type = typename matrix_type::rank_type;
    using extents_type = Std::extents<index_type, matrix_type::static_extent(1)>;
    using value_type = typename matrix_type::value_type;
    using reference = decltype(std::declval<const Storage&>().get()[std::array<index_type,2>{}]);

  private:
    class Iterator
      : public Dune::IteratorFacade<Iterator, std::random_access_iterator_tag,
          std::remove_cvref_t<reference>, reference>
    {
    public:
      constexpr Iterator () = default;
      constexpr Iterator (const RowView* row, index_type j) : row_{row}, j_{j} {}

      constexpr reference operator* () const { return (*row_)[j_]; }

    protected:
      friend Dune::IteratorFacadeAccess;
      constexpr const index_type& baseIterator () const noexcept { return j_; }
      constexpr index_type& baseIterator () noexcept { return j_; }

    private:
      const RowView* row_ = nullptr;
      index_type j_ = 0;
    };

  public:
    using iterator = Iterator;
    using const_iterator = Iterator;

    constexpr RowView (Storage storage, index_type i)
      : storage_{std::move(storage)}
      , i_{i}
    {}

    static constexpr rank_type rank () noexcept { return 1; }
    static constexpr rank_type rank_dynamic () noexcept { return extents_type::rank_dynamic(); }
    static constexpr std::size_t static_extent (rank_type r) noexcept { return extents_type::static_extent(r); }

    constexpr index_type extent ([[maybe_unused]] rank_type r) const { return storage_.get().extent(1); }
    constexpr extents_type extents () const { return extents_type(extent(0)); }

    //! Access the element `j` of the row
    constexpr reference operator[] (index_type j) const
    {
      return storage_.get()[std::array<index_type,2>{i_, j}];
    }

    //! Access the element `{j}` of the row
    template <class Index>
      requires std::is_convertible_v<const Index&, index_type>
    constexpr reference operator[] (const std::array<Index,1>& index) const
    {
      return (*this)[index_type(index[0])];
    }

    //! The number of elements in the row, i.e., `extent(0)`
    constexpr std::size_t size () const { return std::size_t(extent(0)); }

    constexpr iterator begin () const { return iterator{this, 0}; }
    constexpr iterator end () const { return iterator{this, extent(0)}; }

  private:
    Storage storage_;
    index_type i_;
  };

} // end namespace Impl


/**
 * \brief A range of all multi-indices of the index space of `t`, in row-major order.
 *
 * The multi-indices are of type `std::array<index_type,rank>` and can directly be used
 * for element access, e.g., `for (auto const& i : indicesOf(t)) t[i] = 0;`
 */
template <Concept::MultiDimensionalLike T>
constexpr auto indicesOf (const T& t)
{
  const auto& md = asMultiDimensional(t);
  using Extents = typename std::remove_cvref_t<decltype(md)>::extents_type;
  return Impl::IndexSpaceRange<Extents>{Extents(md.extents())};
}

/**
 * \brief A range of all elements of `t`, in row-major order of the multi-indices.
 *
 * The range has a `size()` equal to the number of elements in the index space.
 *
 * \note The range refers to `t`. Thus, `t` must be an lvalue that lives longer than the range.
 */
template <Concept::MultiDimensionalLike T>
constexpr auto entriesOf (T& t)
{
  return transformedRangeView(indicesOf(t),
    [storage = Impl::storeMultiDimensional(t)](const auto& index) -> decltype(auto) {
      return storage.get()[index];
    });
}

//! Disallow ranges over temporaries
template <class T>
  requires (!std::is_lvalue_reference_v<T>)
constexpr auto entriesOf (T&& t) = delete;

/**
 * \brief A range of the rows of a rank-2 object `t`.
 *
 * Each row is a rank-1 view modeling `Concept::MultiDimensional`, with element access
 * `row[j]` and iteration over the row elements. Since the rows are one-dimensional,
 * their `size()` is the number of elements, i.e., the number of columns of `t`.
 *
 * \b Example:
 * \code{.cpp}
   for (auto&& row : rowsOf(matrix))
     for (auto&& entry : row)
       entry = 0;
 * \endcode
 *
 * \note The range refers to `t`. Thus, `t` must be an lvalue that lives longer than the range.
 */
template <Concept::MultiDimensionalLike T>
constexpr auto rowsOf (T& t)
{
  using Storage = decltype(Impl::storeMultiDimensional(t));
  using MD = std::remove_cvref_t<decltype(asMultiDimensional(t))>;
  static_assert(MD::rank() == 2, "rowsOf() requires a rank-2 object");
  using index_type = typename MD::index_type;

  Storage storage = Impl::storeMultiDimensional(t);
  index_type rows = storage.get().extent(0);
  return transformedRangeView(Dune::range(rows),
    [storage](index_type i) { return Impl::RowView<Storage>{storage, i}; });
}

//! Disallow ranges over temporaries
template <class T>
  requires (!std::is_lvalue_reference_v<T>)
constexpr auto rowsOf (T&& t) = delete;

} // end namespace Dune

#endif // DUNE_COMMON_MULTIDIMENSIONAL_HH
