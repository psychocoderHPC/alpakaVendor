Installation
============

alpakaVendor is a header-only library. You can include it directly in your project and enable only the backends needed by the modules you use.

Requirements
------------

- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- CMake 3.25 or later
- alpaka 3.x
- Module- and backend-specific dependencies (see below)

Install alpakaVendor
--------------------

Use alpakaVendor via ``find_package``
+++++++++++++++++++++++++++++++++++++

First, install alpaka:

.. code-block:: bash

   git clone https://github.com/alpaka-group/alpaka.git
   cd alpaka && mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
   cmake --build . --target install

Then install alpakaVendor. The default configuration enables the FFTW and OpenBLAS host backends, so install those
packages first (see `FFT backend dependencies`_ and `BLAS backend dependencies`_) or disable them with
``-DalpakaV_DEP_FFTW=OFF -DalpakaV_DEP_OPENBLAS=OFF``.

.. code-block:: bash

   git clone https://github.com/alpaka-group/alpakaVendor.git
   cd alpakaVendor && mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
   cmake --build . --target install

Use alpakaVendor via ``add_subdirectory``
+++++++++++++++++++++++++++++++++++++++++

Add alpakaVendor as a git submodule:

.. code-block:: bash

   git submodule add https://github.com/alpaka-group/alpakaVendor.git extern/alpakaVendor

Then in your ``CMakeLists.txt``:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.25)
   project("myproject" CXX)

   add_subdirectory(extern/alpaka)
   add_subdirectory(extern/alpakaVendor)

   alpaka_add_executable(${PROJECT_NAME} main.cpp)
   target_link_libraries(${PROJECT_NAME} PUBLIC alpakaVendor::alpakaVendor)

Use alpakaVendor via ``FetchContent``
+++++++++++++++++++++++++++++++++++++

.. code-block:: cmake

   include(FetchContent)

   FetchContent_Declare(
       alpakaVendor
       GIT_REPOSITORY https://github.com/alpaka-group/alpakaVendor.git
       GIT_TAG main
   )
   FetchContent_MakeAvailable(alpakaVendor)

   target_link_libraries(${PROJECT_NAME} PRIVATE alpakaVendor::alpakaVendor)

Modules and dependencies
------------------------

alpakaVendor is organized into modules. Dependencies are enabled through project-wide CMake options, while each module documents its own usage details separately.

Available today
+++++++++++++++

FFT
   Portable C2C, R2C, and C2R transforms in 1D, 2D, and 3D, including batched and in-place real transforms. It can use
   one or more backend libraries depending on your target platforms.

BLAS
   Portable dense linear algebra for alpaka queues and views, including Level-1 vector routines, ``gemv``, ``gemm``,
   ``stridedBatchedGemm``, and ``trsm``. It can use host and accelerator BLAS libraries depending on your target
   platforms.

Planned modules
+++++++++++++++

Parallel primitives
   Planned.

FFT backend dependencies
++++++++++++++++++++++++

Install FFTW for the CPU backend:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt install libfftw3-dev

   # macOS with Homebrew
   brew install fftw

Enable CUDA and cuFFT for the NVIDIA GPU backend:

.. code-block:: bash

   cmake .. -DalpakaV_DEP_CUFFT=ON -Dalpaka_DEP_CUDA=ON -Dalpaka_CUDA_NvidiaGpu=ON

Enable ROCm and rocFFT for the AMD GPU backend:

.. code-block:: bash

   cmake .. -DalpakaV_DEP_ROCFFT=ON -Dalpaka_DEP_HIP=ON -Dalpaka_HIP_AmdGpu=ON

Enable oneAPI and oneMKL DFT for the oneAPI CPU or Intel GPU backend:

.. code-block:: bash

   cmake .. -DCMAKE_CXX_COMPILER=icpx -Dalpaka_DEP_ONEAPI=ON -Dalpaka_ONEAPI_Cpu=ON -DalpakaV_DEP_ONEMKL=ON

Relevant FFT options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``alpakaV_DEP_FFTW``
     - Enable FFTW host backend (default: ON)
   * - ``alpakaV_DEP_CUFFT``
     - Enable cuFFT CUDA backend (default: OFF)
   * - ``alpakaV_DEP_ROCFFT``
     - Enable rocFFT HIP backend (default: OFF)
   * - ``alpakaV_DEP_ONEMKL``
     - Enable oneMKL DFT and BLAS oneAPI backends (default: OFF)

Example CPU-only configuration:

.. code-block:: bash

   cmake .. -DalpakaV_DEP_FFTW=ON -DalpakaV_DEP_CUFFT=OFF -DalpakaV_DEP_ROCFFT=OFF

BLAS backend dependencies
+++++++++++++++++++++++++

Install OpenBLAS for the CPU backend:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt install libopenblas-dev pkg-config

   # macOS with Homebrew
   brew install openblas pkg-config

Enable CUDA and cuBLAS for the NVIDIA GPU backend:

.. code-block:: bash

   cmake .. -DalpakaV_DEP_CUBLAS=ON -Dalpaka_DEP_CUDA=ON -Dalpaka_CUDA_NvidiaGpu=ON

Enable ROCm and rocBLAS for the AMD GPU backend:

.. code-block:: bash

   cmake .. -DalpakaV_DEP_ROCBLAS=ON -Dalpaka_DEP_HIP=ON -Dalpaka_HIP_AmdGpu=ON

Enable oneAPI and oneMKL BLAS for the oneAPI CPU or Intel GPU backend:

.. code-block:: bash

   cmake .. -DCMAKE_CXX_COMPILER=icpx -Dalpaka_DEP_ONEAPI=ON -Dalpaka_ONEAPI_Cpu=ON -DalpakaV_DEP_ONEMKL=ON

Relevant BLAS options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``alpakaV_DEP_OPENBLAS``
     - Enable OpenBLAS / CBLAS host backend (default: ON)
   * - ``alpakaV_DEP_CUBLAS``
     - Enable cuBLAS CUDA backend (default: OFF)
   * - ``alpakaV_DEP_ROCBLAS``
     - Enable rocBLAS HIP backend (default: OFF)
   * - ``alpakaV_DEP_ONEMKL``
     - Enable oneMKL DFT and BLAS oneAPI backends (default: OFF)

Example CPU-only configuration for FFT and BLAS:

.. code-block:: bash

   cmake .. -DalpakaV_DEP_FFTW=ON -DalpakaV_DEP_OPENBLAS=ON \
            -DalpakaV_DEP_CUFFT=OFF -DalpakaV_DEP_CUBLAS=OFF \
            -DalpakaV_DEP_ROCFFT=OFF -DalpakaV_DEP_ROCBLAS=OFF

General options
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``alpakaV_TESTS``
     - Build alpakaVendor tests (default: ON)
   * - ``alpakaV_DOCS``
     - Build documentation and code snippets (default: OFF)

See also
--------

- :doc:`../modules`
- :doc:`../fft/index`
- :doc:`../blas/index`
