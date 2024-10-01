// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include <iostream>

#include <dune/common/densetensor.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/foreachindex.hh>
#include <dune/common/test/testsuite.hh>
#include <utility>

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

template <class D, class B>
void call (const DenseTensorMixin<D,B>& tensor)
{
  // it is possible to call a function that expects the TensorMixin base class
}

void call2 (DenseTensorSpan<const double, Std::dextents<int,0>> tensorspan)
{
  // it is possible to call a function that expects the TensorSpan argument
}
void call2 (DenseTensorSpan<const double, Std::dextents<int,1>> tensorspan)
{
  // it is possible to call a function that expects the TensorSpan argument
}
void call2 (DenseTensorSpan<const double, Std::dextents<int,2>> tensorspan)
{
  // it is possible to call a function that expects the TensorSpan argument
}
void call2 (DenseTensorSpan<const double, Std::dextents<int,3>> tensorspan)
{
  // it is possible to call a function that expects the TensorSpan argument
}

template <class Tensor>
void checkConstructors(Dune::TestSuite& testSuite)
{
  Dune::TestSuite subTestSuite("checkConstructors");

  // default constructor
  std::cout << "default constructor..." << std::endl;
  Tensor tensor;
  checkEqualValue(subTestSuite, tensor, 0.0);

  std::cout << "assignment of value..." << std::endl;
  tensor = 1.0;
  checkEqualValue(subTestSuite, tensor, 1.0);

  // constructor with a default value
  std::cout << "value constructor..." << std::endl;
  Tensor tensor1(1.0);
  checkEqualValue(subTestSuite, tensor1, 1.0);

  // copy/move constructors
  std::cout << "copy constructor..." << std::endl;
  Tensor tensor2{tensor1};
  checkEqual(subTestSuite, tensor1,tensor2);

  Tensor tensor3 = tensor1;
  checkEqual(subTestSuite, tensor1,tensor3);

  std::cout << "move constructor..." << std::endl;
  Tensor tensor4{std::move(tensor2)};
  checkEqual(subTestSuite, tensor1,tensor4);

  Tensor tensor5 = std::move(tensor3);
  checkEqual(subTestSuite, tensor1,tensor5);

  // copy/move assignment operators
  std::cout << "copy assignment..." << std::endl;
  tensor4 = tensor1;
  checkEqual(subTestSuite, tensor1,tensor4);

  std::cout << "move assignment..." << std::endl;
  tensor5 = std::move(tensor4);
  checkEqual(subTestSuite, tensor1,tensor5);

  std::cout << "initializerlist constructor..." << std::endl;
  if constexpr(Tensor::rank() == 1) {
    Tensor tensor6{6,7};
    Tensor tensor7{6.0,7.0};
    Tensor tensor8({6.0,7.0});
    Tensor tensor9{{6.0,7.0}};
    Tensor tensor10 = {6.0,7.0};

    subTestSuite.check(tensor6(0) == 6.0);
    subTestSuite.check(tensor6(1) == 7.0);
    subTestSuite.check(tensor7(0) == 6.0);
    subTestSuite.check(tensor7(1) == 7.0);
    subTestSuite.check(tensor8(0) == 6.0);
    subTestSuite.check(tensor8(1) == 7.0);
    subTestSuite.check(tensor9(0) == 6.0);
    subTestSuite.check(tensor9(1) == 7.0);
    subTestSuite.check(tensor10(0) == 6.0);
    subTestSuite.check(tensor10(1) == 7.0);
  }
  else
  if constexpr(Tensor::rank() == 2) {
    Tensor tensor6{{6,7},{8,9}};
    Tensor tensor7{{6.0,7.0},{8.0,9.0}};
    Tensor tensor8({{6.0,7.0},{8.0,9.0}});
    Tensor tensor9{{{6.0,7.0},{8.0,9.0}}};
    Tensor tensor10 = {{6.0,7.0},{8.0,9.0}};

    subTestSuite.check(tensor6(0,0) == 6.0);
    subTestSuite.check(tensor6(0,1) == 7.0);
    subTestSuite.check(tensor6(1,0) == 8.0);
    subTestSuite.check(tensor6(1,1) == 9.0);
    subTestSuite.check(tensor7(0,0) == 6.0);
    subTestSuite.check(tensor7(0,1) == 7.0);
    subTestSuite.check(tensor7(1,0) == 8.0);
    subTestSuite.check(tensor7(1,1) == 9.0);
    subTestSuite.check(tensor8(0,0) == 6.0);
    subTestSuite.check(tensor8(0,1) == 7.0);
    subTestSuite.check(tensor8(1,0) == 8.0);
    subTestSuite.check(tensor8(1,1) == 9.0);
    subTestSuite.check(tensor9(0,0) == 6.0);
    subTestSuite.check(tensor9(0,1) == 7.0);
    subTestSuite.check(tensor9(1,0) == 8.0);
    subTestSuite.check(tensor9(1,1) == 9.0);
    subTestSuite.check(tensor10(0,0) == 6.0);
    subTestSuite.check(tensor10(0,1) == 7.0);
    subTestSuite.check(tensor10(1,0) == 8.0);
    subTestSuite.check(tensor10(1,1) == 9.0);
  }

  // check whether a function with a TensorMixin argument can be called
  call(tensor);

  {
    DenseTensorSpan span(tensor);
    DenseTensorSpan span2(tensor.toTensorSpan());

    Tensor tensorFromSpan(span);
    Tensor tensorFromSpan2(span2);
  }

  // check whether a function with a TensorSpan argument can be called
  call2(tensor);
  call2(tensor.toTensorSpan());

  {
    // check conversion between dynamic and static extents
    using extents_type = Dune::Std::dextents<std::size_t,Tensor::rank()>;
    [[maybe_unused]] auto dynext = extents_type(tensor.extents());
    [[maybe_unused]] auto ext = (typename Tensor::extents_type)(dynext);

    // check conversion between dynamic and static tensors
    using T = typename Tensor::element_type;
    [[maybe_unused]] auto dyntensor = Dune::DenseTensor{dynext, T(0)};
    [[maybe_unused]] Tensor dynamicToStatic(dyntensor);
    [[maybe_unused]] decltype(dyntensor) staticToDynamic(tensor);
  }

  testSuite.subTest(subTestSuite);
}


template <class Tensor>
void checkAccess(Dune::TestSuite& testSuite)
{
  Dune::TestSuite subTestSuite("checkAccess");

  Tensor tensor(42.0);
  checkEqualValue(subTestSuite, tensor, 42.0);

  if constexpr(Tensor::rank() == 0) {
    subTestSuite.check(tensor[std::array<int,0>{}] == 42.0);
    subTestSuite.check(tensor() == 42.0);
    subTestSuite.check(tensor == 42.0);
    double value = tensor;
    subTestSuite.check(value == 42.0);
  }
  else if constexpr(Tensor::rank() == 1) {
    for (std::size_t i = 0; i < Tensor::static_extent(0); ++i) {
      subTestSuite.check(tensor[std::array{i}] == 42.0);
      subTestSuite.check(tensor(i) == 42.0);
      subTestSuite.check(tensor[i] == 42.0);
      subTestSuite.check(tensor.at(i) == 42.0);
    }
    subTestSuite.checkThrow<Dune::RangeError>([&]{tensor.at(-1);});
    subTestSuite.checkThrow<Dune::RangeError>([&]{tensor.at(tensor.extent(0));});
  }
  else if constexpr(Tensor::rank() == 2) {
    for (std::size_t i = 0; i < Tensor::static_extent(0); ++i) {
      for (std::size_t j = 0; j < Tensor::static_extent(1); ++j) {
        subTestSuite.check(tensor[std::array{i,j}] == 42.0);
        subTestSuite.check(tensor(i,j) == 42.0);
        subTestSuite.check(tensor.at(i,j) == 42.0);
      }
      subTestSuite.checkThrow<Dune::RangeError>([&]{tensor.at(-1,-2);});
      subTestSuite.checkThrow<Dune::RangeError>([&]{tensor.at(tensor.extent(0),tensor.extent(1));});
    }
  }
  else if constexpr(Tensor::rank() == 3) {
    for (std::size_t i = 0; i < Tensor::static_extent(0); ++i) {
      for (std::size_t j = 0; j < Tensor::static_extent(1); ++j) {
        for (std::size_t k = 0; k < Tensor::static_extent(2); ++k) {
          subTestSuite.check(tensor[std::array{i,j,k}] == 42.0);
          subTestSuite.check(tensor(i,j,k) == 42.0);
          subTestSuite.check(tensor.at(i,j,k) == 42.0);
        }
        subTestSuite.checkThrow<Dune::RangeError>([&]{tensor.at(-1,-2,-3);});
        subTestSuite.checkThrow<Dune::RangeError>([&]{tensor.at(tensor.extent(0),tensor.extent(1),tensor.extent(2));});
      }
    }
  }

  testSuite.subTest(subTestSuite);
}


template <class Tensor>
void checkArithmetic(Dune::TestSuite& testSuite)
{
  Dune::TestSuite subTestSuite("checkArithmetic");

  Tensor tensor(1.0);
  Tensor tensor2(2.0);
  checkEqualValue(subTestSuite, tensor, 1.0);
  checkEqualValue(subTestSuite, tensor2, 2.0);

  tensor *= 2.0;
  checkEqualValue(subTestSuite, tensor, 2.0);

  tensor += tensor2;
  checkEqualValue(subTestSuite, tensor, 4.0);

  tensor.axpy(4.0, tensor2); // 12
  checkEqualValue(subTestSuite, tensor, 12.0);

  tensor.aypx(4.0, tensor2); // 50
  checkEqualValue(subTestSuite, tensor, 50.0);

  tensor -= tensor2;
  checkEqualValue(subTestSuite, tensor, 48.0);

  testSuite.subTest(subTestSuite);
}



int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  TestSuite testSuite;

  using Tensor0 = Dune::DenseTensor<double>;
  using Tensor1 = Dune::DenseTensor<double,2>;
  using Tensor2 = Dune::DenseTensor<double,2,2>;
  using Tensor3 = Dune::DenseTensor<double,2,2,2>;

  checkConstructors<Tensor0>(testSuite);
  checkConstructors<Tensor1>(testSuite);
  checkConstructors<Tensor2>(testSuite);
  checkConstructors<Tensor3>(testSuite);

  checkAccess<Tensor0>(testSuite);
  checkAccess<Tensor1>(testSuite);
  checkAccess<Tensor2>(testSuite);
  checkAccess<Tensor3>(testSuite);

  checkArithmetic<Tensor0>(testSuite);
  checkArithmetic<Tensor1>(testSuite);
  checkArithmetic<Tensor2>(testSuite);
  checkArithmetic<Tensor3>(testSuite);

  return testSuite.exit();
}
