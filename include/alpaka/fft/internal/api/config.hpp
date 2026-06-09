/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#if __has_include(<fftw3.h>)
#    include <fftw3.h>
#    define ALPAKAV_HAS_FFTW 1
#else
#    define ALPAKAV_HAS_FFTW 0
#endif

#if __has_include(<cufft.h>)
#    include <cufft.h>
#    define ALPAKAV_HAS_CUFFT 1
#else
#    define ALPAKAV_HAS_CUFFT 0
#endif

#if __has_include(<hipfft/hipfft.h>)
#    include <hipfft/hipfft.h>
#    define ALPAKAV_HAS_HIPFFT 1
#else
#    define ALPAKAV_HAS_HIPFFT 0
#endif
