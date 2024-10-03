// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_FVECTOR_HH
#define DUNE_COMMON_FVECTOR_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <initializer_list>

#include <dune/common/boundschecking.hh>
#include <dune/common/densevector.hh>
#include <dune/common/filledarray.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/math.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/typeutilities.hh>

namespace Dune {

  /** @addtogroup DenseMatVec
      @{
   */

  /*! \file
   * \brief Implements a vector constructed from a given type
     representing a field and a compile-time given size.
   */

  template< class K, int SIZE > class FieldVector;
  template< class K, int SIZE >
  struct DenseMatVecTraits< FieldVector<K,SIZE> >
  {
    typedef FieldVector<K,SIZE> derived_type;
    typedef std::array<K,SIZE> container_type;
    typedef K value_type;
    typedef typename container_type::size_type size_type;
  };

  template< class K, int SIZE >
  struct FieldTraits< FieldVector<K,SIZE> >
  {
    typedef typename FieldTraits<K>::field_type field_type;
    typedef typename FieldTraits<K>::real_type real_type;
  };

  /**
   * @brief TMP to check the size of a DenseVectors statically, if possible.
   *
   * If the implementation type of C is  a FieldVector, we statically check
   * whether its dimension is SIZE.
   * @tparam C The implementation of the other DenseVector
   * @tparam SIZE The size we need assume.
   */
  template<typename C, int SIZE>
  struct IsFieldVectorSizeCorrect
  {
    /**
     * \brief True if C is not of type FieldVector or its dimension
     * is not equal SIZE.
     */
    constexpr static bool value = true;
  };

  template<typename T, int SIZE>
  struct IsFieldVectorSizeCorrect<FieldVector<T,SIZE>,SIZE>
  {
    constexpr static bool value = true;
  };

  template<typename T, int SIZE, int SIZE1>
  struct IsFieldVectorSizeCorrect<FieldVector<T,SIZE1>,SIZE>
  {
    constexpr static bool value = false;
  };


  /** \brief vector space out of a tensor product of fields.
   *
   * \tparam K    the field type (use float, double, complex, etc)
   * \tparam SIZE number of components.
   */
  template< class K, int SIZE >
  class FieldVector :
    public DenseVector< FieldVector<K,SIZE> >
  {
    std::array<K,SIZE> _data;
    typedef DenseVector< FieldVector<K,SIZE> > Base;
  public:
    //! The size of this vector.
    constexpr static int dimension = SIZE;

    typedef typename Base::size_type size_type;
    typedef typename Base::value_type value_type;

    /** \brief The type used for references to the vector entry */
    typedef value_type& reference;

    /** \brief The type used for const references to the vector entry */
    typedef const value_type& const_reference;

    //! Default constructor, making value-initialized vector with all components set to zero
    constexpr FieldVector ()
        noexcept(std::is_nothrow_default_constructible_v<K>)
      : _data{}
    {}

    //! Constructor with a given scalar
    template<class T>
      requires (IsNumber<T>::value && std::constructible_from<K,T>)
    explicit(SIZE > 1) constexpr FieldVector (const T& value)
      : _data{filledArray<SIZE,K>(K(value))}
    {}

    //! Construct from a std::initializer_list
    constexpr FieldVector (const std::initializer_list<K>& l)
      : _data{}
    {
      assert(l.size() == size());
      for (int i = 0; i < SIZE; ++i)
        _data[i] = std::data(l)[i];
    }

    //! Constructor from another dense vector if the elements are assignable to K
    template<class T>
      requires (IsFieldVectorSizeCorrect<T,SIZE>::value &&
        std::assignable_from<K&, decltype(std::declval<const T&>()[0])>)
    FieldVector (const DenseVector<T>& x)
    {
      assert(x.size() == size());
      for (int i = 0; i < SIZE; ++i)
        _data[i] = x[i];
    }

    //! Converting constructor from FieldVector with different element type
    template<class T>
      requires (std::assignable_from<K&, const T&>)
    explicit constexpr FieldVector (const FieldVector<T, SIZE>& x)
        noexcept(std::is_nothrow_assignable_v<K&, const T&>)
    {
      for (int i = 0; i < SIZE; ++i)
        _data[i] = x[i];
    }

    //! Converting constructor with FieldVector of different size (deleted)
    template<class K1, int SIZE1>
      requires (SIZE1 != SIZE)
    FieldVector (const FieldVector<K1, SIZE1>&) = delete;

    //! Copy constructor with default behavior
    constexpr FieldVector (const FieldVector&) = default;


    //! Assignment from another dense vector
    template<class T>
      requires (IsFieldVectorSizeCorrect<T,SIZE>::value &&
        std::assignable_from<K&, decltype(std::declval<const T&>()[0])>)
    FieldVector& operator= (const DenseVector<T>& x)
    {
      assert(x.size() == size());
      for (int i = 0; i < SIZE; ++i)
        _data[i] = x[i];
      return *this;
    }

    //! Assignment operator from scalar
    template <class T>
      requires (IsNumber<T>::value && std::assignable_from<K&, const T&>)
    constexpr FieldVector& operator= (const T& value)
    {
      for (int i = 0; i < SIZE; ++i)
        _data[i] = value;
      return *this;
    }

    //! Converting assignment operator from FieldVector with different element type
    template<class T>
      requires (std::assignable_from<K&, const T&>)
    FieldVector& operator= (const FieldVector<T, SIZE>& x)
        noexcept(std::is_nothrow_assignable_v<K&, const T&>)
    {
      for (int i = 0; i < SIZE; ++i)
        _data[i] = x[i];
      return *this;
    }

    //! Converting assignment operator with FieldVector of different size (deleted)
    template<class K1, int SIZE1>
      requires (SIZE1 != SIZE)
    FieldVector& operator= (const FieldVector<K1, SIZE1>&) = delete;

    //! Copy assignment operator with default behavior
    constexpr FieldVector& operator= (const FieldVector&) = default;

    using Base::operator=;

    //! Obtain the number of elements stored in the vector
    static constexpr size_type size () noexcept { return SIZE; }

    //! Return a reference to the `i`th element
    reference operator[] (size_type i)
    {
      DUNE_ASSERT_BOUNDS(i < size());
      return _data[i];
    }

    //! Return a (const) reference to the `i`th element
    const_reference operator[] (size_type i) const
    {
      DUNE_ASSERT_BOUNDS(i < size());
      return _data[i];
    }

    //! Return pointer to underlying array
    constexpr K* data () noexcept
    {
      return _data.data();
    }

    //! Return pointer to underlying array
    constexpr const K* data () const noexcept
    {
      return _data.data();
    }

    //! Conversion operator
    constexpr operator const_reference () const
        noexcept requires(SIZE == 1)
    {
      return _data[0];
    }


    /// \name Comparison operators
    /// @{

    //! comparison of FieldVectors for equality
    template<class T>
      requires (std::equality_comparable_with<K,T>)
    friend constexpr bool operator== (const FieldVector& a, const FieldVector<T,SIZE>& b)
        noexcept
    {
      return a._data == b._data;
    }

    //! comparing FieldVectors<1> with scalar for equality
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr bool operator== (const FieldVector& a, const T& b)
        noexcept requires(SIZE == 1)
    {
      return a._data[0] == b;
    }

    //! comparing FieldVectors<1> with scalar for equality
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr bool operator== (const T& a, const FieldVector& b)
        noexcept requires(SIZE == 1)
    {
      return a == b._data[0];
    }

    //! (lexicographic) comparison of FieldVectors
    template<class T>
      requires (std::three_way_comparable_with<K,T>)
    friend constexpr auto operator<=> (const FieldVector& a, const FieldVector<T,SIZE>& b)
        noexcept
    {
      return a._data <=> b._data;
    }

    //! (lexicographic) comparison of FieldVectors<1> with scalar
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr auto operator<=> (const FieldVector& a, const T& b)
        noexcept requires(SIZE == 1)
    {
      return a._data[0] <=> b;
    }

    //! (lexicographic) comparison of FieldVectors<1> with scalar
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr auto operator<=> (const T& a, const FieldVector& b)
        noexcept requires(SIZE == 1)
    {
      return a <=> b._data[0];
    }

    /// @}


    /// \name Vector space operations
    /// @{

    //! Vector space multiplication with scalar
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator* (const FieldVector& vector, const T& scalar)
    {
      FieldVector result;
      for (int i = 0; i < SIZE; ++i)
        result[i] = vector[i] * scalar;
      return result;
    }

    //! Vector space multiplication with scalar
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator* (const T& scalar, const FieldVector& vector)
    {
      FieldVector result;
      for (int i = 0; i < SIZE; ++i)
        result[i] = scalar * vector[i];
      return result;
    }

    //! Vector space division by scalar
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator/ (const FieldVector& vector, const T& scalar)
    {
      FieldVector result;
      for (int i = 0; i < SIZE; ++i)
        result[i] = vector[i] / scalar;
      return result;
    }

    //! Binary division, when using FieldVector<K,1> like K
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator/ (const T& a, const FieldVector& b)
        noexcept requires(SIZE == 1)
    {
      return FieldVector{a / b[0]};
    }

    //! Binary addition, when using FieldVector<K,1> like K
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator+ (const FieldVector& a, const T& b)
        noexcept requires(SIZE == 1)
    {
      return FieldVector{a[0] + b};
    }

    //! Binary addition, when using FieldVector<K,1> like K
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator+ (const T& a, const FieldVector& b)
        noexcept requires(SIZE == 1)
    {
      return FieldVector{a + b[0]};
    }

    //! Binary subtraction, when using FieldVector<K,1> like K
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator- (const FieldVector& a, const T& b)
        noexcept requires(SIZE == 1)
    {
      return FieldVector{a[0] - b};
    }

    //! Binary subtraction, when using FieldVector<K,1> like K
    template<class T>
      requires (IsNumber<T>::value)
    friend constexpr FieldVector operator- (const T& a, const FieldVector& b)
        noexcept requires(SIZE == 1)
    {
      return FieldVector{a - b[0]};
    }

    /// @}
  };

  /** \brief Read a FieldVector from an input stream
   *  \relates FieldVector
   *
   *  \note This operator is STL compliant, i.e., the content of v is only
   *        changed if the read operation is successful.
   *
   *  \param[in]  in  std :: istream to read from
   *  \param[out] v   FieldVector to be read
   *
   *  \returns the input stream (in)
   */
  template<class K, int SIZE>
  std::istream& operator>> (std::istream& in, FieldVector<K, SIZE>& v)
  {
    FieldVector<K, SIZE> w;
    for (int i = 0; i < SIZE; ++i)
      in >> w[i];
    if (in)
      v = w;
    return in;
  }

  /* Overloads for common classification functions */
  namespace MathOverloads {

    //! Returns whether all entries are finite
    template<class K, int SIZE>
    auto isFinite (const FieldVector<K,SIZE>& b, PriorityTag<2>, ADLTag)
    {
      bool out = true;
      for (int i = 0; i < SIZE; ++i) {
        out &= Dune::isFinite(b[i]);
      }
      return out;
    }

    //! Returns whether any entry is infinite
    template<class K, int SIZE>
    bool isInf (const FieldVector<K,SIZE>& b, PriorityTag<2>, ADLTag)
    {
      bool out = false;
      for (int i = 0; i < SIZE; ++i) {
        out |= Dune::isInf(b[i]);
      }
      return out;
    }

    //! Returns whether any entry is NaN
    template<class K, int SIZE,
      std::enable_if_t<HasNaN<K>::value, int> = 0>
    bool isNaN (const FieldVector<K,SIZE>& b, PriorityTag<2>, ADLTag)
    {
      bool out = false;
      for (int i = 0; i < SIZE; ++i) {
        out |= Dune::isNaN(b[i]);
      }
      return out;
    }

    //! Returns true if either b or c is NaN
    template<class K,
      std::enable_if_t<HasNaN<K>::value, int> = 0>
    bool isUnordered (const FieldVector<K,1>& b, const FieldVector<K,1>& c,
                      PriorityTag<2>, ADLTag)
    {
      return Dune::isUnordered(b[0],c[0]);
    }

  } // end namespace MathOverloads

  /** @} end documentation */

} // end namespace Dune

#endif // DUNE_COMMON_FVECTOR_HH
