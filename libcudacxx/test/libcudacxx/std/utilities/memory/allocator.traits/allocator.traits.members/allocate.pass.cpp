//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     static constexpr pointer allocate(allocator_type& a, size_type n);
//     ...
// };

#include <cuda/std/__memory>
#include <cuda/std/cstdint>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "incomplete_type_helper.h"

template <class T>
struct A
{
    typedef T value_type;

    __host__ __device__ TEST_CONSTEXPR_CXX20 A() {}

    __host__ __device__ TEST_CONSTEXPR_CXX20 value_type* allocate(cuda::std::size_t n)
    {
        assert(n == 10);
        return &storage;
    }

    value_type storage;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
    {
        A<int> a;
        assert(cuda::std::allocator_traits<A<int> >::allocate(a, 10) == &a.storage);
    }
    {
        typedef A<IncompleteHolder*> Alloc;
        Alloc a;
        assert(cuda::std::allocator_traits<Alloc>::allocate(a, 10) == &a.storage);
    }

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 2020
    static_assert(test());
#endif // TEST_STD_VER >= 2020
    return 0;
}
