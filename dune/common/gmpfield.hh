// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_GMPFIELD_HH
#define DUNE_COMMON_GMPFIELD_HH

/** \file
 * \brief Wrapper for the GNU MPF(R) multiprecision floating point library
 */

#include <iostream>
#include <string>
#include <type_traits>

#if HAVE_GMP || DOXYGEN

#if HAVE_MPFR
#include <mpreal.h>
#elif HAVE_GMPXX
#include <gmpxx.h> // fallback implementation
#endif

#include <dune/common/promotiontraits.hh>
#include <dune/common/typetraits.hh>

namespace Dune
{
  /**
   * \ingroup Numbers
   * \brief Number class for high precision floating point number using the MPF(R) library mpreal implementation
   */
  template< unsigned int precision >
  class GMPField
#if HAVE_MPFR
    : public mpfr::mpreal
#elif HAVE_GMPXX
    : public mpf_class
#endif
  {
#if HAVE_MPFR
    using Base = mpfr::mpreal;
    using Prec = mp_prec_t;
#elif HAVE_GMPXX
    using Base = mpf_class;
    using Prec = mp_bitcnt_t;
#endif

  static_assert(precision > 0 && precision < std::numeric_limits<Prec>::max());

  public:
    //! default constructor, initialize to zero.
    GMPField ()
      : Base(0, Prec(precision))
    {}

    /**
     * \brief initialize from a string
     * \note this is the only reliable way to initialize with higher precision values
     */
    explicit GMPField (const char* str)
      : Base(str, Prec(precision))
    {}

    /**
     * \brief initialize from a string
     * \note this is the only reliable way to initialize with higher precision values
     */
    explicit GMPField (const std::string& str)
      : Base(str, Prec(precision))
    {}

    //! initialize from from mpreal value.
    GMPField (const Base& v)
      : Base(v)
    {}

    //! initialize from a compatible scalar type.
    template <class T,
      std::enable_if_t<std::is_arithmetic_v<T>,int> = 0>
    GMPField (const T& v)
      : Base(v, Prec(precision))
    {}

    GMPField (const GMPField&) = default;
    GMPField (GMPField&&) = default;

    GMPField& operator= (GMPField const&) = default;
    GMPField& operator= (GMPField&&) = default;

    //! type conversion operators
    operator double () const
    {
#if HAVE_MPFR
      return this->toDouble();
#elif HAVE_GMPXX
      return this->get_d();
#endif
    }
  };

  template <unsigned int precision>
  struct IsNumber<GMPField<precision>>
    : public std::true_type {};

  template <unsigned int precision1, unsigned int precision2>
  struct PromotionTraits<GMPField<precision1>, GMPField<precision2>>
  {
    using PromotedType = GMPField<(precision1 > precision2 ? precision1 : precision2)>;
  };

  template <unsigned int precision>
  struct PromotionTraits<GMPField<precision>,GMPField<precision>>
  {
    using PromotedType = GMPField<precision>;
  };

  template <unsigned int precision, class T>
  struct PromotionTraits<GMPField<precision>, T>
  {
    using PromotedType = GMPField<std::max<unsigned int>(8*sizeof(T), precision)>;
  };

  template <class T, unsigned int precision>
  struct PromotionTraits<T, GMPField<precision>>
  {
    using PromotedType = GMPField<std::max<unsigned int>(8*sizeof(T), precision)>;
  };
}

namespace std
{
  /// Specialization of numeric_limits for known precision width
#if HAVE_MPFR
  template <unsigned int precision>
  inline void swap(Dune::GMPField<precision>& x, Dune::GMPField<precision>& y)
  {
      return mpfr::swap(x, y);
  }

  template <unsigned int precision>
  class numeric_limits<Dune::GMPField<precision>>
      : public numeric_limits<mpfr::mpreal>
  {
    using type = Dune::GMPField<precision>;

  public:
    inline static type min () { return mpfr::minval(precision); }
    inline static type max () {  return  mpfr::maxval(precision); }
    inline static type lowest () { return -mpfr::maxval(precision); }
    inline static type epsilon () { return  mpfr::machine_epsilon(precision); }

    inline static type round_error ()
    {
      mp_rnd_t r = mpfr::mpreal::get_default_rnd();

      if(r == GMP_RNDN)  return mpfr::mpreal(0.5, precision);
      else               return mpfr::mpreal(1.0, precision);
    }

    inline static int digits () { return int(precision); }
    inline static int digits10 () { return mpfr::bits2digits(precision); }
    inline static int max_digits10 () { return mpfr::bits2digits(precision); }
  };
#elif HAVE_GMPXX
  template <unsigned int precision>
  inline void swap(Dune::GMPField<precision>& x, Dune::GMPField<precision>& y)
  {
      return x.swap(y);
  }

  // NOTE that the specialization of std::numeric_limits for mpf_class is not usable
  // since all entries are simply set to 0. Thus, we do not provide a specialization
  // for the wrapper class GMPField, which would lead to wrong results and could not
  // be detected by `std::numeric_limit<GMPField<...>>::is_specialized.
#endif

} // end namespace std

#endif // HAVE_GMP

#endif // DUNE_COMMON_GMPFIELD_HH
