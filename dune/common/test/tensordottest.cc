// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include <dune/common/tensor.hh>
#include <dune/common/tensordot.hh>
#include <dune/common/tensorspan.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>

using namespace Dune;
int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  TestSuite testSuite;

  auto dTensor0 = Dune::Tensor<double>{};
  auto dTensor1 = Dune::Tensor<double,Dune::dynamic>{2};
  auto dTensor2 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic>{2,2};
  auto dTensor3 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic,Dune::dynamic>{2,2,2};

  auto fTensor0 = Dune::Tensor<double>{};
  auto fTensor1 = Dune::Tensor<double,2>{};
  auto fTensor2 = Dune::Tensor<double,2,2>{};
  auto fTensor3 = Dune::Tensor<double,2,2,2>{};

  // test dynamic tensors
  {
    auto dScalar = tensordot<0>(dTensor0,dTensor0);
    testSuite.check(dScalar.rank() == 0);
  }

  {
    auto dVec0 = tensordot<0>(dTensor0,dTensor1);
    testSuite.check(dVec0.rank() == 1);
    testSuite.check(dVec0.extent(0) == 2);

    auto dVec1 = tensordot<0>(dTensor1,dTensor0);
    testSuite.check(dVec1.rank() == 1);
    testSuite.check(dVec1.extent(0) == 2);

    auto dMat = tensordot<0>(dTensor1,dTensor1);
    testSuite.check(dMat.rank() == 2);
    testSuite.check(dMat.extent(0) == 2);
    testSuite.check(dMat.extent(1) == 2);

    auto dScalar = tensordot<1>(dTensor1,dTensor1);
    testSuite.check(dScalar.rank() == 0);
  }

  {
    auto dMat0 = tensordot<0>(dTensor0,dTensor2);
    testSuite.check(dMat0.rank() == 2);
    testSuite.check(dMat0.extent(0) == 2);
    testSuite.check(dMat0.extent(1) == 2);

    auto dMat1 = tensordot<0>(dTensor2,dTensor0);
    testSuite.check(dMat1.rank() == 2);
    testSuite.check(dMat1.extent(0) == 2);
    testSuite.check(dMat1.extent(1) == 2);

    auto dVec0 = tensordot<1>(dTensor1,dTensor2);
    testSuite.check(dVec0.rank() == 1);
    testSuite.check(dVec0.extent(0) == 2);

    auto dVec1 = tensordot<1>(dTensor2,dTensor1);
    testSuite.check(dVec1.rank() == 1);
    testSuite.check(dVec1.extent(0) == 2);

    auto dTen0 = tensordot<0>(dTensor1,dTensor2);
    testSuite.check(dTen0.rank() == 3);
    testSuite.check(dTen0.extent(0) == 2);
    testSuite.check(dTen0.extent(1) == 2);
    testSuite.check(dTen0.extent(2) == 2);

    auto dTen1 = tensordot<0>(dTensor2,dTensor1);
    testSuite.check(dTen1.rank() == 3);
    testSuite.check(dTen1.extent(0) == 2);
    testSuite.check(dTen1.extent(1) == 2);
    testSuite.check(dTen1.extent(2) == 2);

    auto dScalar = tensordot<2>(dTensor2,dTensor2);
    testSuite.check(dScalar.rank() == 0);

    auto dMat2 = tensordot<1>(dTensor2,dTensor2);
    testSuite.check(dMat2.rank() == 2);
    testSuite.check(dMat2.extent(0) == 2);
    testSuite.check(dMat2.extent(1) == 2);

    auto dTen2 = tensordot<0>(dTensor2,dTensor2);
    testSuite.check(dTen2.rank() == 4);
    testSuite.check(dTen2.extent(0) == 2);
    testSuite.check(dTen2.extent(1) == 2);
    testSuite.check(dTen2.extent(2) == 2);
    testSuite.check(dTen2.extent(3) == 2);
  }

  // test mixed static/dynamic tensors
  {
    auto dScalar = tensordot<0>(fTensor0,dTensor0);
    testSuite.check(dScalar.rank() == 0);
  }

  {
    auto dVec0 = tensordot<0>(fTensor0,dTensor1);
    testSuite.check(dVec0.rank() == 1);
    testSuite.check(dVec0.extent(0) == 2);

    auto dVec1 = tensordot<0>(fTensor1,dTensor0);
    testSuite.check(dVec1.rank() == 1);
    testSuite.check(dVec1.extent(0) == 2);

    auto dMat = tensordot<0>(fTensor1,dTensor1);
    testSuite.check(dMat.rank() == 2);
    testSuite.check(dMat.extent(0) == 2);
    testSuite.check(dMat.extent(1) == 2);

    auto dScalar = tensordot<1>(fTensor1,dTensor1);
    testSuite.check(dScalar.rank() == 0);
  }

  {
    auto dMat0 = tensordot<0>(fTensor0,dTensor2);
    testSuite.check(dMat0.rank() == 2);
    testSuite.check(dMat0.extent(0) == 2);
    testSuite.check(dMat0.extent(1) == 2);

    auto dMat1 = tensordot<0>(fTensor2,dTensor0);
    testSuite.check(dMat1.rank() == 2);
    testSuite.check(dMat1.extent(0) == 2);
    testSuite.check(dMat1.extent(1) == 2);

    auto dVec0 = tensordot<1>(fTensor1,dTensor2);
    testSuite.check(dVec0.rank() == 1);
    testSuite.check(dVec0.extent(0) == 2);

    auto dVec1 = tensordot<1>(fTensor2,dTensor1);
    testSuite.check(dVec1.rank() == 1);
    testSuite.check(dVec1.extent(0) == 2);

    auto dTen0 = tensordot<0>(fTensor1,dTensor2);
    testSuite.check(dTen0.rank() == 3);
    testSuite.check(dTen0.extent(0) == 2);
    testSuite.check(dTen0.extent(1) == 2);
    testSuite.check(dTen0.extent(2) == 2);

    auto dTen1 = tensordot<0>(fTensor2,dTensor1);
    testSuite.check(dTen1.rank() == 3);
    testSuite.check(dTen1.extent(0) == 2);
    testSuite.check(dTen1.extent(1) == 2);
    testSuite.check(dTen1.extent(2) == 2);

    auto dScalar = tensordot<2>(fTensor2,dTensor2);
    testSuite.check(dScalar.rank() == 0);

    auto dMat2 = tensordot<1>(fTensor2,dTensor2);
    testSuite.check(dMat2.rank() == 2);
    testSuite.check(dMat2.extent(0) == 2);
    testSuite.check(dMat2.extent(1) == 2);

    auto dTen2 = tensordot<0>(fTensor2,dTensor2);
    testSuite.check(dTen2.rank() == 4);
    testSuite.check(dTen2.extent(0) == 2);
    testSuite.check(dTen2.extent(1) == 2);
    testSuite.check(dTen2.extent(2) == 2);
    testSuite.check(dTen2.extent(3) == 2);
  }

  // test interaction with TensorSpan
  {
    auto dMat0 = tensordot<1>(fTensor2,dTensor2.toTensorSpan());
    auto dMat1 = tensordot<1>(fTensor2.toTensorSpan(),dTensor2);
    auto dMat2 = tensordot<1>(fTensor2.toTensorSpan(),dTensor2.toTensorSpan());
  }

  // test output-tensor
  {
    tensordotOut<1>(dTensor2,dTensor2,fTensor2);
    tensordotOut<1>(fTensor2,fTensor2,dTensor2);
  }

  // test arbitrary index contractions
  {
    auto dScalar = tensordot(fTensor2,std::index_sequence<0,1>{}, dTensor2,std::index_sequence<1,0>{});
    testSuite.check(dScalar.rank() == 0);
    tensordotOut(fTensor2,std::index_sequence<0,1>{}, dTensor2,std::index_sequence<1,0>{}, dScalar);

    auto dMat = tensordot(fTensor2,std::index_sequence<0>{}, dTensor2,std::index_sequence<0>{});
    testSuite.check(dMat.rank() == 2);
    tensordotOut(fTensor2,std::index_sequence<0>{}, dTensor2,std::index_sequence<0>{}, dMat);

    auto dTen = tensordot(fTensor2,std::index_sequence<>{}, dTensor2,std::index_sequence<>{});
    testSuite.check(dTen.rank() == 4);
    tensordotOut(fTensor2,std::index_sequence<>{}, dTensor2,std::index_sequence<>{}, dTen);
  }

  return testSuite.exit();
}
