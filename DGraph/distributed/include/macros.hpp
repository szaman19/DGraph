/**
 * Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the LBANN Research Team (B. Van Essen, et al.) listed in
 * the CONTRIBUTORS file. See the top-level LICENSE file for details.
 *
 * LLNL-CODE-697807.
 * All rights reserved.
 *
 * This file is part of LBANN: Livermore Big Artificial Neural Network
 * Toolkit. For details, see http://software.llnl.gov/LBANN or
 * https://github.com/LBANN and https://github.com/LLNL/LBANN.
 *
 * SPDX-License-Identifier: (Apache-2.0)
 */
#pragma once

#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
// CUDA 13 switched __cudaLaunch to a two-argument macro; shim legacy callers.
#include <crt/host_runtime.h>
#if defined(__cudaLaunch) && !defined(DGRAPH_WRAP_CUDA_LAUNCH)
#define DGRAPH_WRAP_CUDA_LAUNCH
#define __cudaLaunch_two_arg __cudaLaunch
#undef __cudaLaunch
#define __cudaLaunch(fun) __cudaLaunch_two_arg(fun, 0)
#endif
#endif

#define CUDACHECK(cmd)                          \
  do                                            \
  {                                             \
    cudaError_t e = cmd;                        \
    if (e != cudaSuccess)                       \
    {                                           \
      printf("Failed: Cuda error %s:%d '%s'\n", \
             __FILE__, __LINE__,                \
             cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                       \
    }                                           \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  do                    \
  {                     \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
  } while (0)
