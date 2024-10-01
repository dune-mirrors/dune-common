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

  // test tensor mixin methods
  {
    Concept::Tensor auto d = fTensor2 * dTensor2;
    Concept::Tensor auto d3 = fTensor32 * dTensor2;
    Concept::Tensor auto d34 = fTensor2 * dTensor234;
  }

  {
    Concept::Tensor auto d = fTensor2.dot(dTensor2);
    Concept::Tensor auto d3 = fTensor32.dot(dTensor2);
    Concept::Tensor auto d34 = fTensor2.dot(dTensor234);
  }

  {
    Concept::Tensor auto d = dTensor23.ddot(fTensor23);
    Concept::Tensor auto d4 = dTensor23.ddot(fTensor234);
  }

  {
    fTensor32.mv(dTensor2, fTensor3);
    fTensor23.mtv(dTensor2, fTensor3);
    fTensor23.mhv(dTensor2, fTensor3);

    fTensor32.umv(dTensor2, fTensor3);
    fTensor23.umtv(dTensor2, fTensor3);
    fTensor23.umhv(dTensor2, fTensor3);

    fTensor32.mmv(dTensor2, fTensor3);
    fTensor23.mmtv(dTensor2, fTensor3);
    fTensor23.mmhv(dTensor2, fTensor3);

    fTensor32.usmv(2.0, dTensor2, fTensor3);
    fTensor23.usmtv(2.0, dTensor2, fTensor3);
    fTensor23.usmhv(2.0, dTensor2, fTensor3);
  }

  {
    std::floating_point auto x1 = dTensor23.inner(fTensor23);
    std::floating_point auto x2 = dTensor23.frobenius_norm2();
    std::floating_point auto x3 = dTensor23.frobenius_norm();

    std::floating_point auto x4 = fTensor3.two_norm2();
    std::floating_point auto x5 = fTensor3.two_norm();
  }

  {
    // a tensor of complex number initialized with initilizer-lists
    Dune::Tensor<std::complex<double>, 2,2> cTensor22{
      {std::complex<double>(1.0,2.0), std::complex<double>(-1.0,0.5)},
      {std::complex<double>(-1.0,0.4), std::complex<double>( 1.0,3.0)}
    };

    // construct the complex numbers using brace-init lists
    Dune::Tensor<std::complex<double>, 2,2> cTensor22_{
      { {1.0,2.0},  {-1.0,0.5} },
      { {-1.0,0.4}, {1.0,3.0}  }
    };

    // the dot product and Hermitian product are different
    auto x1 = cTensor22 * cTensor22;
    auto x2 = cTensor22.dot(cTensor22);
    testSuite.check(x1 != x2);
  }

  return testSuite.exit();
}
