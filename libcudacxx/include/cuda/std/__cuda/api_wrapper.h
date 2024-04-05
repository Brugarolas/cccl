//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__STD__CUDA_API_WRAPPER_H
#define _CUDA__STD__CUDA_API_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/detail/libcxx/include/__exception/terminate.h>
#include <cuda/std/detail/libcxx/include/stdexcept>
#if !defined(_CCCL_COMPILER_NVRTC)
#  include <cstdio>
#endif // !_CCCL_COMPILER_NVRTC

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

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
class cuda_error : public _CUDA_VSTD_NOVERSION::runtime_error
{
public:
  _CCCL_HOST cuda_error(::cudaError_t __status, const char* __msg) noexcept
      : _CUDA_VSTD_NOVERSION::runtime_error("")
  {
    ::snprintf(
      const_cast<char*>(this->what()), _CUDA_VSTD::__libcpp_refstring::__length, "cudaError %d: %s", __status, __msg);
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif //_CUDA__STD__CUDA_API_WRAPPER_H
