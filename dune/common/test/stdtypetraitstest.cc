// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <dune/common/std/type_traits.hh>

#include <dune/common/test/testsuite.hh>


int main()
{
  Dune::TestSuite test;

  // Check negation
  {
    test.check(Dune::Std::negation<std::true_type>::value == false)
      << "Dune::Std::negation of std::true_type is not false";
    test.check(Dune::Std::negation<std::false_type>::value == true)
      << "Dune::Std::negation of std::false_type is not true";
    test.check(Dune::Std::negation<Dune::Std::negation<std::true_type>>::value == true)
      << "Double Dune::Std::negation is not the identity";
    test.check(Dune::Std::negation<Dune::Std::negation<std::false_type>>::value == false)
      << "Double Dune::Std::negation is not the identity";
  }

  return test.exit();
}
