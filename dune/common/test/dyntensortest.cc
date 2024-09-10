// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include <array>

#include <dune/common/filledarray.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/tensor.hh>
#include <dune/common/tensorspan.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/test/foreachindex.hh>
#include <type_traits>

using namespace Dune;

template <class Tensor1, class Tensor2>
void checkEqual(Dune::TestSuite& testSuite, Tensor1 const& a, Tensor2 const& b)
{
  Dune::TestSuite subTestSuite("checkEqual");
  subTestSuite.require(a.extents() == b.extents());
  Dune::forEachIndex(a.extents(), [&](auto index) {
    subTestSuite.check(a[index] == b[index]);
  });
  testSuite.subTest(subTestSuite);
}

template <class Tensor>
void checkEqualValue(Dune::TestSuite& testSuite, Tensor const& a, typename Tensor::value_type const& value)
{
  Dune::TestSuite subTestSuite("checkEqualValue");
  Dune::forEachIndex(a.extents(), [&](auto index) {
    subTestSuite.check(a[index] == value);
  });
  testSuite.subTest(subTestSuite);
}


template <class Tensor>
void checkConstructors(Dune::TestSuite& testSuite)
{
  Dune::TestSuite subTestSuite("checkConstructors");

  using T = typename Tensor::element_type;
  using extents_type = typename Tensor::extents_type;
  extents_type ext{Dune::filledArray<Tensor::rank()>(3)};

  // default constructor
  Tensor tensor0;
  subTestSuite.check(tensor0.size() == 0 || tensor0.rank() == 0);

  Tensor tensor(ext);
  checkEqualValue(subTestSuite, tensor, T(0));

  if constexpr(Tensor::rank() > 0) {
    tensor0.resize(ext, T(0));
    subTestSuite.check(tensor0.size() == tensor.size());
  }

#if 0 // not yet implemented
  tensor = 1.0;
  checkEqualValue(subTestSuite, tensor, 1.0);
#endif

  // constructor with a default value
  Tensor tensor1(ext, T(1));
  checkEqualValue(subTestSuite, tensor1, T(1));

  // copy/move constructors
  Tensor tensor2{tensor1};
  checkEqual(subTestSuite, tensor1,tensor2);

  Tensor tensor3 = tensor1;
  checkEqual(subTestSuite, tensor1,tensor3);

  Tensor tensor4{std::move(tensor2)};
  checkEqual(subTestSuite, tensor1,tensor4);

  Tensor tensor5 = std::move(tensor3);
  checkEqual(subTestSuite, tensor1,tensor5);

  // copy/move assignment operators
  tensor4 = tensor1;
  checkEqual(subTestSuite, tensor1,tensor4);

  tensor5 = std::move(tensor4);
  checkEqual(subTestSuite, tensor1,tensor5);

  extents_type ext2{Dune::filledArray<Tensor::rank()>(2)};
  if constexpr(Tensor::rank() == 1) {
    if constexpr (std::is_floating_point_v<T>) {
      Tensor tensor6(ext2, {6,7});
      subTestSuite.check(tensor6(0) == T(6.0));
      subTestSuite.check(tensor6(1) == T(7.0));
    }

    Tensor tensor7{ext2, {T(6.0),T(7.0)}};
    subTestSuite.check(tensor7(0) == T(6.0));
    subTestSuite.check(tensor7(1) == T(7.0));
  }
  else
  if constexpr(Tensor::rank() == 2) {
    if constexpr (std::is_floating_point_v<T>) {
      Tensor tensor6(ext2, {{6,7},{8,9}});
      subTestSuite.check(tensor6(0,0) == T(6.0));
      subTestSuite.check(tensor6(0,1) == T(7.0));
      subTestSuite.check(tensor6(1,0) == T(8.0));
      subTestSuite.check(tensor6(1,1) == T(9.0));
    }

    Tensor tensor7(ext2, {{T(6.0),T(7.0)},{T(8.0),T(9.0)}});
    Tensor tensor8a(std::array<int,2>{2,2}, {{T(6.0),T(7.0)},{T(8.0),T(9.0)}});
    Tensor tensor8b(std::array<unsigned int,2>{2u,2u}, {{T(6.0),T(7.0)},{T(8.0),T(9.0)}});

    subTestSuite.check(tensor7(0,0) == T(6.0));
    subTestSuite.check(tensor7(0,1) == T(7.0));
    subTestSuite.check(tensor7(1,0) == T(8.0));
    subTestSuite.check(tensor7(1,1) == T(9.0));
  }


  Tensor tensor9(tensor1.toTensorSpan());

  { // check deduction guides

    Dune::Tensor t1{tensor1.extents(), T{}};
    Dune::Tensor t2{tensor1.mapping(), T{}};
    Dune::Tensor t3{tensor1.toTensorSpan()};
  }

  testSuite.subTest(subTestSuite);
}


template <class Tensor>
void checkAccess(Dune::TestSuite& testSuite)
{
  Dune::TestSuite subTestSuite("checkAccess");

  using T = typename Tensor::element_type;
  using extents_type = typename Tensor::extents_type;
  using index_type = typename Tensor::index_type;
  extents_type ext{Dune::filledArray<Tensor::rank()>(3)};

  Tensor tensor(ext, T(42.0));
  checkEqualValue(subTestSuite, tensor, T(42.0));

  if constexpr(Tensor::rank() == 0) {
    subTestSuite.check(tensor[std::array<std::size_t,0>{}] == T(42.0));
    subTestSuite.check(tensor() == T(42.0));
    subTestSuite.check(tensor.at() == T(42.0));
    subTestSuite.check(tensor == T(42.0));
    double value = tensor;
    subTestSuite.check(value == T(42.0));
  }
  else if constexpr(Tensor::rank() == 1) {
    for (index_type i = 0; i < tensor.extent(0); ++i) {
      subTestSuite.check(tensor[std::array{i}] == T(42.0));
      subTestSuite.check(tensor[i] == T(42.0));
      subTestSuite.check(tensor(i) == T(42.0));
      subTestSuite.check(tensor.at(i) == T(42.0));
    }
  }
  else if constexpr(Tensor::rank() == 2) {
    for (index_type i = 0; i < tensor.extent(0); ++i) {
      for (index_type j = 0; j < tensor.extent(1); ++j) {
        subTestSuite.check(tensor[std::array{i,j}] == T(42.0));
        subTestSuite.check(tensor(i,j) == T(42.0));
        subTestSuite.check(tensor.at(i,j) == T(42.0));
      }
    }
  }
  else if constexpr(Tensor::rank() == 3) {
    for (index_type i = 0; i < tensor.extent(0); ++i) {
      for (index_type j = 0; j < tensor.extent(1); ++j) {
        for (index_type k = 0; k < tensor.extent(2); ++k) {
          subTestSuite.check(tensor[std::array{i,j,k}] == T(42.0));
          subTestSuite.check(tensor(i,j,k) == T(42.0));
          subTestSuite.check(tensor.at(i,j,k) == T(42.0));
        }
      }
    }
  }

  testSuite.subTest(subTestSuite);
}


template <class Tensor>
void checkArithmetic(Dune::TestSuite& testSuite)
{
  Dune::TestSuite subTestSuite("checkArithmetic");

  using T = typename Tensor::element_type;
  using extents_type = typename Tensor::extents_type;
  extents_type ext{Dune::filledArray<Tensor::rank()>(3)};

  Tensor tensor(ext, T(1));
  Tensor tensor2(ext, T(2));
  checkEqualValue(subTestSuite, tensor, T(1));
  checkEqualValue(subTestSuite, tensor2, T(2));

#if 0 // not yet implemented
  if constexpr(std::is_floating_point_v<T>) {
    tensor *= 2.0;
    checkEqualValue(subTestSuite, tensor, T(2));

    tensor += tensor2;
    checkEqualValue(subTestSuite, tensor, T(4));

    tensor.axpy(4.0, tensor2); // 12
    checkEqualValue(subTestSuite, tensor, T(12));

    tensor.aypx(4.0, tensor2); // 50
    checkEqualValue(subTestSuite, tensor, T(50.0));

    tensor -= tensor2;
    checkEqualValue(subTestSuite, tensor, T(48.0));
  }
#endif

  testSuite.subTest(subTestSuite);
}



int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  TestSuite testSuite;

  using Tensor0 = Dune::Tensor<double>;
  using Tensor1 = Dune::Tensor<double,Dune::dynamic>;
  using Tensor2 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic>;
  using Tensor3 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic,Dune::dynamic>;

  // using Tensor4 = Dune::Tensor<bool,Dune::dynamic>;
  // using Tensor5 = Dune::Tensor<bool,Dune::dynamic,Dune::dynamic>;

  checkConstructors<Tensor0>(testSuite);
  checkConstructors<Tensor1>(testSuite);
  checkConstructors<Tensor2>(testSuite);
  checkConstructors<Tensor3>(testSuite);
  // checkConstructors<Tensor4>(testSuite);
  // checkConstructors<Tensor5>(testSuite);

  checkAccess<Tensor0>(testSuite);
  checkAccess<Tensor1>(testSuite);
  checkAccess<Tensor2>(testSuite);
  checkAccess<Tensor3>(testSuite);
  // checkAccess<Tensor4>(testSuite);
  // checkAccess<Tensor5>(testSuite);

  checkArithmetic<Tensor0>(testSuite);
  checkArithmetic<Tensor1>(testSuite);
  checkArithmetic<Tensor2>(testSuite);
  checkArithmetic<Tensor3>(testSuite);
  // checkArithmetic<Tensor4>(testSuite);
  // checkArithmetic<Tensor5>(testSuite);

  return testSuite.exit();
}
