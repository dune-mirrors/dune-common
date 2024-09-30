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

  auto dTensor = Dune::Tensor<double>{};
  auto dTensor2 = Dune::Tensor<double,Dune::dynamic>{2};
  auto dTensor3 = Dune::Tensor<double,Dune::dynamic>{3};
  auto dTensor23 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic>{2,3};
  auto dTensor32 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic>{3,2};
  auto dTensor234 = Dune::Tensor<double,Dune::dynamic,Dune::dynamic,Dune::dynamic>{2,3,4};

  auto fTensor = Dune::Tensor<double>{};
  auto fTensor2 = Dune::Tensor<double,2>{};
  auto fTensor3 = Dune::Tensor<double,3>{};
  auto fTensor23 = Dune::Tensor<double,2,3>{};
  auto fTensor32 = Dune::Tensor<double,3,2>{};
  auto fTensor234 = Dune::Tensor<double,2,3,4>{};

  // test dynamic tensors
  {
    auto d = tensordot<0>(dTensor,dTensor);
    testSuite.check(d.rank() == 0);
  }

  {
    auto d = tensordot<1>(dTensor2,dTensor2);
    testSuite.check(d.rank() == 0);

    auto d2 = tensordot<0>(dTensor,dTensor2);
    testSuite.check(d2.rank() == 1);
    testSuite.check(d2.extent(0) == 2);

    auto d3 = tensordot<0>(dTensor3,dTensor);
    testSuite.check(d3.rank() == 1);
    testSuite.check(d3.extent(0) == 3);

    auto d23 = tensordot<0>(dTensor2,dTensor3);
    testSuite.check(d23.rank() == 2);
    testSuite.check(d23.extent(0) == 2);
    testSuite.check(d23.extent(1) == 3);
  }

  {
    auto d = tensordot<2>(dTensor23,dTensor23);
    testSuite.check(d.rank() == 0);

    auto d2 = tensordot<1>(dTensor23,dTensor3);
    testSuite.check(d2.rank() == 1);
    testSuite.check(d2.extent(0) == 2);

    auto d3 = tensordot<1>(dTensor2,dTensor23);
    testSuite.check(d3.rank() == 1);
    testSuite.check(d3.extent(0) == 3);

    auto d22 = tensordot<1>(dTensor23,dTensor32);
    testSuite.check(d22.rank() == 2);
    testSuite.check(d22.extent(0) == 2);
    testSuite.check(d22.extent(1) == 2);

    auto d23 = tensordot<0>(dTensor,dTensor23);
    testSuite.check(d23.rank() == 2);
    testSuite.check(d23.extent(0) == 2);
    testSuite.check(d23.extent(1) == 3);

    auto d32 = tensordot<0>(dTensor32,dTensor);
    testSuite.check(d32.rank() == 2);
    testSuite.check(d32.extent(0) == 3);
    testSuite.check(d32.extent(1) == 2);

    auto d223 = tensordot<0>(dTensor2,dTensor23);
    testSuite.check(d223.rank() == 3);
    testSuite.check(d223.extent(0) == 2);
    testSuite.check(d223.extent(1) == 2);
    testSuite.check(d223.extent(2) == 3);

    auto d233 = tensordot<0>(dTensor23,dTensor3);
    testSuite.check(d233.rank() == 3);
    testSuite.check(d233.extent(0) == 2);
    testSuite.check(d233.extent(1) == 3);
    testSuite.check(d233.extent(2) == 3);

    auto d2332 = tensordot<0>(dTensor23,dTensor32);
    testSuite.check(d2332.rank() == 4);
    testSuite.check(d2332.extent(0) == 2);
    testSuite.check(d2332.extent(1) == 3);
    testSuite.check(d2332.extent(2) == 3);
    testSuite.check(d2332.extent(3) == 2);
  }

  // test mixed static/dynamic tensors
  {
    auto d = tensordot<0>(fTensor,dTensor);
    testSuite.check(d.rank() == 0);
  }

  {
    auto d = tensordot<1>(fTensor2,dTensor2);
    testSuite.check(d.rank() == 0);

    auto d2 = tensordot<0>(fTensor,dTensor2);
    testSuite.check(d2.rank() == 1);
    testSuite.check(d2.extent(0) == 2);

    auto d3 = tensordot<0>(fTensor3,dTensor);
    testSuite.check(d3.rank() == 1);
    testSuite.check(d3.extent(0) == 3);

    auto d23 = tensordot<0>(fTensor2,dTensor3);
    testSuite.check(d23.rank() == 2);
    testSuite.check(d23.extent(0) == 2);
    testSuite.check(d23.extent(1) == 3);
  }

  {
    auto d = tensordot<2>(fTensor23,dTensor23);
    testSuite.check(d.rank() == 0);

    auto d2 = tensordot<1>(fTensor23,dTensor3);
    testSuite.check(d2.rank() == 1);
    testSuite.check(d2.extent(0) == 2);

    auto d3 = tensordot<1>(fTensor2,dTensor23);
    testSuite.check(d3.rank() == 1);
    testSuite.check(d3.extent(0) == 3);

    auto d22 = tensordot<1>(fTensor23,dTensor32);
    testSuite.check(d22.rank() == 2);
    testSuite.check(d22.extent(0) == 2);
    testSuite.check(d22.extent(1) == 2);

    auto d23 = tensordot<0>(fTensor,dTensor23);
    testSuite.check(d23.rank() == 2);
    testSuite.check(d23.extent(0) == 2);
    testSuite.check(d23.extent(1) == 3);

    auto d32 = tensordot<0>(fTensor32,dTensor);
    testSuite.check(d32.rank() == 2);
    testSuite.check(d32.extent(0) == 3);
    testSuite.check(d32.extent(1) == 2);

    auto d223 = tensordot<0>(fTensor2,dTensor23);
    testSuite.check(d223.rank() == 3);
    testSuite.check(d223.extent(0) == 2);
    testSuite.check(d223.extent(1) == 2);
    testSuite.check(d223.extent(2) == 3);

    auto d233 = tensordot<0>(fTensor23,dTensor3);
    testSuite.check(d233.rank() == 3);
    testSuite.check(d233.extent(0) == 2);
    testSuite.check(d233.extent(1) == 3);
    testSuite.check(d233.extent(2) == 3);

    auto d2332 = tensordot<0>(fTensor23,dTensor32);
    testSuite.check(d2332.rank() == 4);
    testSuite.check(d2332.extent(0) == 2);
    testSuite.check(d2332.extent(1) == 3);
    testSuite.check(d2332.extent(2) == 3);
    testSuite.check(d2332.extent(3) == 2);
  }

  // test interaction with TensorSpan
  {
    auto dMat0 = tensordot<1>(fTensor23,dTensor32.toTensorSpan());
    auto dMat1 = tensordot<1>(fTensor23.toTensorSpan(),dTensor32);
    auto dMat2 = tensordot<1>(fTensor23.toTensorSpan(),dTensor32.toTensorSpan());
  }

  // test output-tensor
  {
    auto fTensor22 = Tensor<double,2,2>{};
    tensordotOut<1>(dTensor23,dTensor32,fTensor22);
    auto dTensor22 = Tensor<double,Dune::dynamic,Dune::dynamic>{2,2};
    tensordotOut<1>(fTensor23,fTensor32,dTensor22);
  }

  // test arbitrary index contractions
  {
    auto d = tensordot(fTensor23,std::index_sequence<0,1>{}, dTensor32,std::index_sequence<1,0>{});
    testSuite.check(d.rank() == 0);
    tensordotOut(fTensor23,std::index_sequence<0,1>{}, dTensor32,std::index_sequence<1,0>{}, d);

    auto d33 = tensordot(fTensor23,std::index_sequence<0>{}, dTensor23,std::index_sequence<0>{});
    testSuite.check(d33.rank() == 2);
    testSuite.check(d33.extent(0) == 3);
    testSuite.check(d33.extent(1) == 3);
    tensordotOut(fTensor23,std::index_sequence<0>{}, dTensor23,std::index_sequence<0>{}, d33);

    auto d2323 = tensordot(fTensor23,std::index_sequence<>{}, dTensor23,std::index_sequence<>{});
    testSuite.check(d2323.rank() == 4);
    testSuite.check(d2323.extent(0) == 2);
    testSuite.check(d2323.extent(1) == 3);
    testSuite.check(d2323.extent(2) == 2);
    testSuite.check(d2323.extent(3) == 3);
    tensordotOut(fTensor23,std::index_sequence<>{}, dTensor23,std::index_sequence<>{}, d2323);
  }

  return testSuite.exit();
}
