//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_API_WRAPPER_H
#define _CUDA__MEMORY_RESOURCE_CUDA_API_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/detail/libcxx/include/__exception/terminate.h>

#ifndef _LIBCUDACXX_NO_EXCEPTIONS
#  define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)         \
    const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    switch (__status)                                  \
    {                                                  \
      case ::cudaSuccess:                              \
        break;                                         \
      default:                                         \
        ::cudaGetLastError();                          \
        throw ::cuda::cuda_error{__status, _MSG};      \
    }
#else // ^^^ !_LIBCUDACXX_NO_EXCEPTIONS ^^^ / vvv _LIBCUDACXX_NO_EXCEPTIONS vvv
#  define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)         \
    const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    switch (__status)                                  \
    {                                                  \
      case ::cudaSuccess:                              \
        break;                                         \
      default:                                         \
        ::cudaGetLastError();                          \
        _CUDA_VSTD_NOVERSION::terminate();             \
    }
#endif // _LIBCUDACXX_NO_EXCEPTIONS

#define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)      \
  const ::cudaError_t __status = _NAME(__VA_ARGS__); \
  _LIBCUDACXX_ASSERT(__status == cudaSuccess, _MSG); \
  (void) __status;

#endif //_CUDA__MEMORY_RESOURCE_CUDA_API_WRAPPER_H
