//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ALL_OF_H
#define _LIBCUDACXX___CONCEPTS_ALL_OF_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/disjunction.h"
#include "../__type_traits/is_same.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2017
template <bool... _Preds>
_LIBCUDACXX_CONCEPT __all_of = (_Preds && ...);
#elif _CCCL_STD_VER >= 2014
template <bool... _Preds>
struct __all_of_helper;

template <bool... _Preds>
_LIBCUDACXX_CONCEPT __all_of = _IsSame<__all_of_helper<_Preds...>, __all_of_helper<((void) _Preds, true)...>>::value;
#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_ALL_OF_H
