// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <config.h>

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/multidimensional.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/concepts/multidimensional.hh>
#include <dune/common/concepts/archetypes/multidimensional.hh>
#include <dune/common/std/extents.hh>
#include <dune/common/std/mdarray.hh>
#include <dune/common/std/mdspan.hh>
#include <dune/common/test/testsuite.hh>

using namespace Dune;

// Generic algorithms restricted to the interface of `Concept::MultiDimensional`.
// They are instantiated with the archetype to check this restriction.

template <Concept::MultiDimensionalLike T>
double sumOfEntries (const T& t)
{
  double s = 0;
  for (const auto& e : entriesOf(t))
    s += e;
  return s;
}

template <Concept::MultiDimensionalLike T>
double sumOfEntriesByIndex (const T& t)
{
  const auto& md = asMultiDimensional(t);
  double s = 0;
  for (const auto& i : indicesOf(md))
    s += md[i];
  return s;
}

template <Concept::MultiDimensionalLike T>
void fillRows (T& t)
{
  for (auto&& row : rowsOf(t)) {
    double j = 0;
    for (auto&& e : row)
      e = j++;
  }
}


// Check the mdspan-like shape and element access of dense vectors and matrices
template <class V>
void checkDenseVector (TestSuite& testSuite, V& v)
{
  TestSuite subTestSuite("checkDenseVector");

  static_assert(Concept::MultiDimensionalWithRank<V,1>);
  static_assert(V::rank() == 1);
  subTestSuite.check(v.extent(0) == v.size());
  subTestSuite.check(v.extents().extent(0) == v.size());
  if constexpr(V::rank_dynamic() == 0)
    subTestSuite.check(V::static_extent(0) == v.size());

  for (std::size_t i = 0; i < v.size(); ++i) {
    v[std::array{i}] = i;
    subTestSuite.check(v[i] == i);
    std::array<int,1> index{int(i)};
    subTestSuite.check(std::as_const(v)[std::span{index}] == i);
  }

  testSuite.subTest(subTestSuite);
}

template <class M>
void checkDenseMatrix (TestSuite& testSuite, M& m)
{
  TestSuite subTestSuite("checkDenseMatrix");

  static_assert(Concept::MultiDimensionalWithRank<M,2>);
  static_assert(M::rank() == 2);
  subTestSuite.check(m.extent(0) == m.N());
  subTestSuite.check(m.extent(1) == m.M());
  subTestSuite.check(m.extents().extent(0) == m.N());
  subTestSuite.check(m.extents().extent(1) == m.M());
  if constexpr(M::rank_dynamic() == 0) {
    subTestSuite.check(M::static_extent(0) == m.N());
    subTestSuite.check(M::static_extent(1) == m.M());
  }

  for (std::size_t i = 0; i < m.N(); ++i) {
    for (std::size_t j = 0; j < m.M(); ++j) {
      m[std::array{i,j}] = 10*i + j;
      subTestSuite.check(m[i][j] == 10*i + j);
      std::array<int,2> indices{int(i),int(j)};
      subTestSuite.check(std::as_const(m)[std::span{indices}] == 10*i + j);
#if __cpp_multidimensional_subscript >= 202110L
      subTestSuite.check(m[i,j] == 10*i + j);
#endif
    }
  }

  testSuite.subTest(subTestSuite);
}


int main ()
{
  TestSuite testSuite;

  // models of the concept
  static_assert(Concept::MultiDimensional<FieldVector<double,3>>);
  static_assert(Concept::MultiDimensional<FieldMatrix<double,2,3>>);
  static_assert(Concept::MultiDimensional<DynamicVector<double>>);
  static_assert(Concept::MultiDimensional<DynamicMatrix<double>>);
  static_assert(Concept::MultiDimensional<Impl::ScalarVectorView<double>>);
  static_assert(Concept::MultiDimensional<Impl::ScalarMatrixView<double>>);
  static_assert(Concept::MultiDimensional<Std::mdspan<double,Std::dims<3>>>);
  static_assert(Concept::MultiDimensional<Std::mdarray<double,Std::extents<int,2,2>>>);
  static_assert(Concept::MultiDimensional<Concept::Archetypes::MultiDimensional<const double,Std::dims<2>>>);

  static_assert(not Concept::MultiDimensional<double>);
  static_assert(not Concept::MultiDimensional<std::vector<double>>);
  static_assert(not Concept::MultiDimensional<std::array<double,3>>);
  static_assert(not Concept::MultiDimensional<FieldMatrix<double,2,3>&>);

  static_assert(Concept::MultiDimensionalLike<std::vector<double>&>);
  static_assert(Concept::MultiDimensionalLike<const std::array<double,3>&>);
  static_assert(Concept::MultiDimensionalLike<std::span<double>>);
  static_assert(Concept::MultiDimensionalLike<std::vector<double>>);
  static_assert(not Concept::MultiDimensionalLike<double>);

  // static extents of the dense types
  static_assert(std::is_same_v<FieldVector<double,3>::extents_type, Std::extents<std::size_t,3>>);
  static_assert(std::is_same_v<FieldMatrix<double,2,3>::extents_type, Std::extents<std::size_t,2,3>>);
  static_assert(std::is_same_v<DynamicVector<double>::extents_type, Std::dims<1>>);
  static_assert(std::is_same_v<DynamicMatrix<double>::extents_type, Std::dims<2>>);

  { // mdspan-like interface of dense vectors and matrices
    FieldVector<double,3> fv;
    checkDenseVector(testSuite, fv);
    DynamicVector<double> dv(4);
    checkDenseVector(testSuite, dv);
    double s = 0;
    Impl::ScalarVectorView<double> sv(&s);
    checkDenseVector(testSuite, sv);

    FieldMatrix<double,2,3> fm;
    checkDenseMatrix(testSuite, fm);
    DynamicMatrix<double> dm(3,2);
    checkDenseMatrix(testSuite, dm);
    Impl::ScalarMatrixView<double> sm(&s);
    checkDenseMatrix(testSuite, sm);
  }

  { // asMultiDimensional
    FieldMatrix<double,2,3> fm{{1,2,3},{4,5,6}};
    testSuite.check(&asMultiDimensional(fm) == &fm, "asMultiDimensional(lvalue) returns reference");
    static_assert(std::is_same_v<decltype(asMultiDimensional(FieldVector<double,2>{})), FieldVector<double,2>>);

    std::vector<double> vec{1,2,3,4};
    auto mvec = asMultiDimensional(vec);
    static_assert(std::is_same_v<decltype(mvec), Std::mdspan<double,Std::dims<1>>>);
    testSuite.check(mvec.extent(0) == 4);
    testSuite.check(&mvec[std::array{2}] == &vec[2]);

    const std::array<double,3> arr{1,2,3};
    auto marr = asMultiDimensional(arr);
    static_assert(std::is_same_v<decltype(marr), Std::mdspan<const double,Std::extents<std::size_t,3>>>);
    testSuite.check(marr[std::array{1}] == 2);

    auto mspan = asMultiDimensional(std::span<double,4>{vec});
    static_assert(decltype(mspan)::static_extent(0) == 4);
  }

  { // indexSpaceSize differs from size() for matrices
    FieldMatrix<double,2,3> fm;
    testSuite.check(fm.size() == 2);
    testSuite.check(indexSpaceSize(fm) == 6);
    testSuite.check(indexSpaceSize(DynamicMatrix<double>(4,5)) == 20);
    testSuite.check(indexSpaceSize(std::vector<double>(7)) == 7);
    testSuite.check(indexSpaceSize(Std::mdarray<double,Std::extents<int,2,3,4>>{}) == 24);
  }

  { // indicesOf
    std::vector<std::array<std::size_t,3>> indices;
    for (const auto& i : indicesOf(Std::mdarray<double,Std::extents<std::size_t,2,1,3>>{}))
      indices.push_back(i);
    testSuite.check(indices.size() == 6);
    testSuite.check(indices.front() == std::array<std::size_t,3>{0,0,0});
    testSuite.check(indices[1] == std::array<std::size_t,3>{0,0,1});
    testSuite.check(indices[3] == std::array<std::size_t,3>{1,0,0});
    testSuite.check(indices.back() == std::array<std::size_t,3>{1,0,2});

    std::size_t count = 0;
    for ([[maybe_unused]] const auto& i : indicesOf(Std::mdarray<double,Std::extents<int>>{}))
      ++count;
    testSuite.check(count == 1, "rank-0 index space has one index");

    count = 0;
    for ([[maybe_unused]] const auto& i : indicesOf(DynamicMatrix<double>(0,3)))
      ++count;
    testSuite.check(count == 0, "empty index space has no index");
  }

  { // entriesOf and rowsOf
    FieldMatrix<double,2,3> fm{{1,2,3},{4,5,6}};
    testSuite.check(entriesOf(fm).size() == 6);
    testSuite.check(sumOfEntries(fm) == 21);
    testSuite.check(sumOfEntriesByIndex(fm) == 21);

    std::vector<double> vec{1,2,3,4};
    testSuite.check(sumOfEntries(vec) == 10);
    testSuite.check(sumOfEntriesByIndex(vec) == 10);

    for (auto& e : entriesOf(vec))
      e = 1;
    testSuite.check(sumOfEntries(vec) == 4);

    std::size_t numRows = 0;
    for (auto&& row : rowsOf(fm)) {
      static_assert(Concept::MultiDimensionalWithRank<std::remove_cvref_t<decltype(row)>,1>);
      testSuite.check(row.size() == 3);
      testSuite.check(row.extent(0) == 3);
      ++numRows;
    }
    testSuite.check(numRows == 2);

    fillRows(fm);
    testSuite.check(fm == FieldMatrix<double,2,3>{{0,1,2},{0,1,2}});

    DynamicMatrix<double> dm(3,4);
    fillRows(dm);
    testSuite.check(sumOfEntries(dm) == 3*6);

    std::array<double,12> data{};
    Std::mdspan<double,Std::dims<2>> span(data.data(), 4, 3);
    fillRows(span);
    testSuite.check(sumOfEntries(span) == 4*3);
  }

  { // generic algorithms instantiated with the archetype
    std::array<double,6> data{1,2,3,4,5,6};
    Concept::Archetypes::MultiDimensional<double,Std::extents<int,2,3>> a{data.data()};
    testSuite.check(sumOfEntries(a) == 21);
    testSuite.check(sumOfEntriesByIndex(a) == 21);
    testSuite.check(indexSpaceSize(a) == 6);
    fillRows(a);
    testSuite.check(sumOfEntries(a) == 6);

    Concept::Archetypes::MultiDimensional<const double,Std::dims<1>> b{data.data(), Std::dims<1>(6)};
    testSuite.check(sumOfEntries(b) == 6);
  }

  return testSuite.exit();
}
