//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<sized_sentinel_for<I> I2, sized_sentinel_for<I> S2>
//   requires sized_sentinel_for<S, I2>
// friend iter_difference_t<I2> operator-(
//   const common_iterator& x, const common_iterator<I2, S2>& y);

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sized_sentinel_type<int*>>(iter1);
    auto commonSent1 = cuda::std::common_iterator<decltype(iter1), sized_sentinel_type<int*>>(sized_sentinel_type<int*>{buffer + 8});
    assert(commonIter1 - commonSent1 == -8);
    assert(commonSent1 - commonIter1 == 8);
    assert(commonIter1 - commonIter1 == 0);
    assert(commonSent1 - commonSent1 == 0);
  }
  {
    auto iter1 = simple_iterator<int*>(buffer);
    auto iter2 = comparable_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sized_sentinel_type<int*>>(iter1);
    auto commonIter2 = cuda::std::common_iterator<decltype(iter2), sized_sentinel_type<int*>>(iter2);

    assert(commonIter1 - commonIter2 == 0);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    const auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sized_sentinel_type<int*>>(iter1);
    const auto commonSent1 = cuda::std::common_iterator<decltype(iter1), sized_sentinel_type<int*>>(sized_sentinel_type<int*>{buffer + 8});
    assert(commonIter1 - commonSent1 == -8);
    assert(commonSent1 - commonIter1 == 8);
    assert(commonIter1 - commonIter1 == 0);
    assert(commonSent1 - commonSent1 == 0);
  }
  {
    auto iter1 = simple_iterator<int*>(buffer);
    auto iter2 = comparable_iterator<int*>(buffer);
    const auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sized_sentinel_type<int*>>(iter1);
    const auto commonIter2 = cuda::std::common_iterator<decltype(iter2), sized_sentinel_type<int*>>(iter2);

    assert(commonIter1 - commonIter2 == 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
