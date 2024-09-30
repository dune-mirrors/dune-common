// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_COMMON_STD_FORCE_INLINE_HH
#define DUNE_COMMON_STD_FORCE_INLINE_HH

#if __has_include(<version>)
#include <version>
#endif

/**
 * \file Provide the  macro `DUNE_FORCE_INLINE` that expands to `[[always_inline]]`
 * or similar depending on the compiler (version). This code is based on
 * https://meghprkh.github.io/blog/posts/c++-force-inline/
 *
 * The effect of DUNE_FORCE_INLINE is typically as follows:
 * - Ignore `-fno-inline`
 * - Ignore the inlining limits hence inlining the function regardless. It also
 *   inlines functions with alloca calls, which `inline` keyword never does.
 * - Not produce an external definition of a function with external linkage if
 *   marked with `always_inline`.
 **/

#if defined(__clang__)
  // Clang does not generate an error for non-inlinable `always_inline` functions.
  #define DUNE_FORCE_INLINE __attribute__((always_inline)) inline

#elif defined(__GNUC__)
  // GCC would generate an error if it cant always_inline
  #define DUNE_FORCE_INLINE __attribute__((always_inline)) inline

#elif defined(_MSC_VER)
  // MSVC generates a warning for for non-inlinable `__forceinline` functions. But
  // only if compiled with any "inline expansion" optimization (`/Ob<n>`). This is
  // present with `/O1` or `/O2`. We promote this warning to an error.
  #pragma warning(error: 4714)
  #define DUNE_FORCE_INLINE [[msvc::forceinline]]

#else
  #define DUNE_FORCE_INLINE inline
#endif

#endif // DUNE_COMMON_STD_FORCE_INLINE_HH
