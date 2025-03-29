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

#include <dune/common/math.hh>
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
      return this->get_d();
    }

#if HAVE_MPFR
    //! return a double representation
    double get_d () const
    {
      return this->toDouble();
    }
#endif
  };

} // end namespace Dune

#if HAVE_GMPXX
// add a missing function for gmpxx
inline mpf_class round (const mpf_class& value)
{
  mpf_class fvalue = floor(value);
  mpf_class cvalue = ceil(value);
  return (value - fvalue) < (cvalue - value) ? fvalue : cvalue;
}
#endif

namespace Dune
{
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

#if HAVE_MPFR
  template< unsigned int precision >
  struct MathematicalConstants<GMPField<precision>>
  {
    using T = GMPField<precision>;
    static const T e ()
    {
      return mpfr::const_euler(mp_prec_t(precision));
    }

    static const T pi ()
    {
      return mpfr::const_pi(mp_prec_t(precision));
    }
  };
#elif HAVE_GMPXX
  template< unsigned int precision >
  struct MathematicalConstants<GMPField<precision>>
  {
    static_assert(precision < 3319u, "Mathematical constants for GMPField defined up to a precision 3318."); // 1000 digits10

    using T = GMPField<precision>;
    static const T e ()
    {
      static const T e =  T("2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320030599218174135966290435729003342952605956307381323286279434907632338298807531952510190115738341879307021540891499348841675092447614606680822648001684774118537423454424371075390777449920695517027618386062613313845830007520449338265602976067371132007093287091274437470472306969772093101416928368190255151086574637721112523897844250569536967707854499699679468644549059879316368892300987931277361782154249992295763514822082698951936680331825288693984964651058209392398294887933203625094431173012381970684161403970198376793206832823764648042953118023287825098194558153017567173613320698112509961818815930416903515988885193458072738667385894228792284998920868058257492796104841984443634632449684875602336248270419786232090021609902353043699418491463140934317381436405462531520961836908887070167683964243781405927145635490613031072085103837505101157477041718986106873969655212671546889570350");
      return e;
    }

    static const T pi ()
    {
      static const T pi = T("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201");
      return pi;
    }
  };
#endif

} // end namespace Dune

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

    static constexpr int bits2digits (int prec)
    {
      constexpr double LOG10_2 = 0.301029995663981195213738894724493;
      return int(prec * LOG10_2);
    }

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

    static constexpr int digits = int(precision);
    static constexpr int digits10 = bits2digits(precision);
    static constexpr int max_digits10 = bits2digits(precision);
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
