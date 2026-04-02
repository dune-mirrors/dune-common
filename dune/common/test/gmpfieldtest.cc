// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <limits>

#include <dune/common/deprecated.hh>
#include <dune/common/float_cmp.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/gmpfield.hh>
#include <dune/common/math.hh>
#include <dune/common/test/testsuite.hh>

template <class T>
struct Comparator
{
  Comparator(T tol)
    : tol_(tol)
  {}

  bool operator()(T x, T y) const
  {
    return Dune::FloatCmp::eq<T,Dune::FloatCmp::absolute>(x, y, tol_);
  }

private:
  T tol_;
};

int main ()
{
  Dune::TestSuite test;
  using G = Dune::GMPField<128>;

  auto cmp = [](const G& x, const G& y, const G& tol) {
    return Dune::FloatCmp::eq<G,Dune::FloatCmp::absolute>(x, y, tol);
  };

  const auto tol = G::fromString("1e-12");

  using F = double;
  const auto e0 = Dune::StandardMathematicalConstants<F>::e();
  const auto pi0 = Dune::StandardMathematicalConstants<F>::pi();
  const auto e = Dune::StandardMathematicalConstants<G>::e();
  const auto pi = Dune::StandardMathematicalConstants<G>::pi();

  using std::abs;
  test.check(abs(e - G(e0)) < tol, "mathematical constant e");
  test.check(abs(pi - G(pi0)) < tol, "mathematical constant pi");

  auto fromString = G::fromString("0.5");
  test.check(cmp(fromString, G(0.5), tol), "non-deprecated string construction");

  G x1 = int(3);
  G x2 = float(3.25);
  G x3 = double(3.25);
  G x4 = (long double)(3.25);

  test.check(cmp(x1, G(3), tol), "implicit arithmetic construction");
  test.check(cmp(x2, G(3.25), tol), "float construction");
  test.check(cmp(x3, G(3.25), tol), "double construction");
  test.check(cmp(x4, G(3.25), tol), "long double construction");

  DUNE_NO_DEPRECATED_BEGIN
  G deprecatedString = "0.5";
  double deprecatedDouble = x3;
  DUNE_NO_DEPRECATED_END
  test.check(cmp(deprecatedString, fromString, tol), "deprecated implicit string construction");
  test.check(abs(G(deprecatedDouble) - x3) < tol, "deprecated implicit double conversion");

  Dune::FieldVector<G,3> v{1,2,3}, x;
  Dune::FieldMatrix<G,3,3> M{ {1,2,3}, {2,3,4}, {3,4,6} }, A;
  Dune::FieldMatrix<G,3,3> M2{ {1,2,3}, {2,3,4}, {3,4,7} };

  auto y1 = v.one_norm();
  test.check(cmp(y1, G(6.0), tol), "vec.one_norm()");

  auto y2 = v.two_norm();
  test.check(cmp(y2, sqrt(G(14.0)), tol), "vec.two_norm()");

  auto y3 = v.infinity_norm();
  test.check(cmp(y3, G(3.0), tol), "vec.infinity_norm()");

  M.mv(v, x);   // x = M*v
  M.mtv(v, x);  // x = M^T*v
  M.umv(v, x);  // x+= M*v
  M.umtv(v, x); // x+= M^T*v
  M.mmv(v, x);  // x-= M*v
  M.mmtv(v, x); // x-= M^T*v

  auto w1 = M.infinity_norm();
  test.check(cmp(w1, G(13.0), tol), "mat.infinity_norm()");

  auto w2 = M.determinant();
  test.check(cmp(w2, G(-1.0), tol), "mat.determinant()");

  M.solve(v, x);  // x = M^(-1)*v

  [[maybe_unused]] auto M3 = M.leftmultiplyany(M2);
  [[maybe_unused]] auto M4 = M.rightmultiplyany(M2);

  using namespace Dune::FMatrixHelp;
  invertMatrix(M,A);

  test.check(cmp(abs(G{-1}), G{1}, tol), "abs");
  test.check(cmp(fabs(G{-1}), G{1}, tol), "fabs");
  test.check(cmp(pow(G{2}, G{3}), G{8}, tol), "pow");
  test.check(cmp(cbrt(G{0.5*0.5*0.5}), G{0.5}, tol), "cbrt");
  test.check(cmp(sqrt(G{4}), G{2}, tol), "sqrt");
  test.check(cmp(log(exp(G{1.5})), G{1.5}, tol), "log(exp)");

#if HAVE_MPFR
  Comparator<G> tightcmp{std::numeric_limits<G>::epsilon() * 8};
  Comparator<G> weakcmp{cbrt(std::numeric_limits<G>::epsilon())};

  test.check(tightcmp(std::numeric_limits<G>::epsilon(), mpfr::machine_epsilon(128)), "numeric_limits::epsilon()");
  test.check(tightcmp(std::numeric_limits<G>::min(), mpfr::minval(128)), "numeric_limits::min()");
  test.check(tightcmp(std::numeric_limits<G>::max(), mpfr::maxval(128)), "numeric_limits::max()");

  test.check(tightcmp(cos(acos(G{0.5})),G{0.5}), "cos(acos)");
  test.check(tightcmp(cosh(acosh(G{1.5})),G{1.5}), "cosh(acosh)");
  test.check(tightcmp(sin(asin(G{0.5})),G{0.5}), "sin(asin)");
  test.check(tightcmp(sinh(asinh(G{0.5})),G{0.5}), "sinh(asinh)");
  test.check(tightcmp(tan(atan(G{0.5})),G{0.5}), "tan(atan)");
  test.check(tightcmp(atan2(G{1},G{2}), atan(G{0.5})), "atan2");
  test.check(tightcmp(tanh(atanh(G{0.5})),G{0.5}), "tanh(atanh)");
  test.check(tightcmp(fma(G{0.5},G{0.4},G{1.8}),(G{0.5} * G{0.4}) + G{1.8}), "fma");
  test.check(tightcmp(fmax(G{0.6},G{0.4}),G{0.6}), "fmax");
  test.check(tightcmp(fmin(G{0.6},G{0.4}),G{0.4}), "fmin");
  test.check(tightcmp(hypot(G{1.6}, G{2.3}), sqrt(G{1.6}*G{1.6} + G{2.3}*G{2.3})), "hypot");
  test.check(tightcmp(rint(G{2.3}),G{2}), "rint");
  test.check(tightcmp(llround(G{2.3}),(long long int)(2)), "llround");
  test.check(tightcmp(lround(G{2.3}),(long int)(2)), "lround");
  test.check(tightcmp(round(G{2.3}),G{2}), "round");
  test.check(tightcmp(trunc(G{2.7}),G{2}), "trunc");
  test.check(tightcmp(ceil(G{1.6}),G{2}), "ceil");
  test.check(tightcmp(floor(G{1.6}),G{1}), "floor");
  test.check(tightcmp(exp(G{0.2}+G{0.4}), exp(G{0.2})*exp(G{0.4})), "exp");
  test.check(tightcmp(expm1(G{0.6}),exp(G{0.6})-G{1}), "expm1");
  test.check(tightcmp(log10(G{1000}),G{3}), "log10");
  test.check(tightcmp(log2(G{8}),G{3}), "log2");
  test.check(tightcmp(log1p(G{1.6}),log(G{1} + G{1.6})), "log1p");
  test.check(weakcmp(fmod(G{5.1},G{3}),G{2.1}), "fmod");
  test.check(weakcmp(remainder(G{5.1},G{3}),G{-0.9}), "remainder");
  test.check(tightcmp(pow(pi,G{3}),pow(pi,3)), "pow(pi)");
  test.check(tightcmp(erf(G{0}),G{0}), "erf");
  test.check(tightcmp(erfc(G{0.6}), G{1}-erf(G{0.6})), "erfc");
  test.check(tightcmp(lgamma(G{3}),log(G{2})), "lgamma");
  test.check(tightcmp(tgamma(G{3}),G{2}), "tgamma");
#endif

  return test.exit();
}
