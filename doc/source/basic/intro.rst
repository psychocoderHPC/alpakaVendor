Introduction
============

alpakaVendor is a collection of header-only C++20 libraries that provide portable, accelerated math operations on top of the `alpaka <https://github.com/alpaka-group/alpaka>`_ accelerator abstraction library.

About alpakaVendor
------------------

alpakaVendor aims to provide a unified interface for common math operations across CPU and GPU backends. Each module wraps a specific domain of functionality:

**FFT**
   Portable FFT abstraction mapping to vendor-specific implementations (FFTW, cuFFT, rocFFT, oneMKL DFT). Supports C2C, R2C, and C2R transforms in 1D, 2D, and 3D, with batched and in-place modes.

**BLAS**
   (Planned) Portable BLAS interface for linear algebra operations.

**Parallel primitives**
   (Planned) Data-parallel building blocks such as elementwise transforms, sorting, scans, and reductions.

Design principles
-----------------

The library follows these design principles:

1. **Common subset, not maximum features**: Each module exposes a clean common subset that maps well to all supported backends.

2. **Backend-agnostic code**: Write your math code once, run it on any supported backend by changing only the CMake configuration.

3. **Type safety**: Strong typing with compile-time dimension and type selection.

4. **RAII-based resources**: Plans and resources are RAII objects that own backend handles and automatically clean up.

See :doc:`../modules` for the module-oriented documentation layout.

Supported backends
------------------

- **FFTW** (CPU): The fastest Fourier transform in the West. Supports float and double precision.
- **cuFFT** (NVIDIA GPU): NVIDIA's CUDA FFT library. Requires CUDA toolkit.
- **rocFFT** (AMD GPU): AMD's HIP FFT library.
- **oneMKL** (Intel GPU): Intel's oneAPI Math Kernel Library. (Planned)

Requirements
------------

- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- CMake 3.25 or later
- alpaka 3.x
- Backend-specific dependencies (see :doc:`install`)
