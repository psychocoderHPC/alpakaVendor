FFT Quickstart
===============

This page shows backend-agnostic FFT examples. The C++ code works the same for every available backend; only the CMake configuration decides which backends are enabled.

Basic C2C transform
-------------------

The examples are instantiated for all available backends via alpaka's backend generator:

.. literalinclude:: ../../../doc/code/quickstart_c2c.cpp
   :language: C++
   :start-after: //! [quickstart-c2c-backends]
   :end-before: //! [quickstart-c2c-backends]

The actual FFT code stays backend-agnostic:

.. literalinclude:: ../../../doc/code/quickstart_c2c.cpp
   :language: C++
   :start-after: //! [quickstart-c2c-core]
   :end-before: //! [quickstart-c2c-core]

.. raw:: html

   <details>
   <summary><strong>Full source: quickstart_c2c.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/quickstart_c2c.cpp
   :language: C++
   :caption: quickstart_c2c.cpp

.. raw:: html

   </details>

Basic R2C transform
-------------------

Real-to-complex transforms compute only the non-redundant half of the spectrum:

.. literalinclude:: ../../../doc/code/quickstart_r2c.cpp
   :language: C++
   :start-after: //! [quickstart-r2c-core]
   :end-before: //! [quickstart-r2c-core]

.. raw:: html

   <details>
   <summary><strong>Full source: quickstart_r2c.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/quickstart_r2c.cpp
   :language: C++
   :caption: quickstart_r2c.cpp

.. raw:: html

   </details>

In-place R2C/C2R roundtrip
--------------------------

For in-place transforms, use ``SharedBufferFFT`` which manages padding automatically:

.. literalinclude:: ../../../doc/code/quickstart_inplace.cpp
   :language: C++
   :start-after: //! [quickstart-inplace-core]
   :end-before: //! [quickstart-inplace-core]

.. raw:: html

   <details>
   <summary><strong>Full source: quickstart_inplace.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/quickstart_inplace.cpp
   :language: C++
   :caption: quickstart_inplace.cpp

.. raw:: html

   </details>

Compiling and running
---------------------

Enable host FFT support:

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DalpakaV_DEP_FFTW=ON -DalpakaV_DEP_CUFFT=OFF -DalpakaV_DEP_ROCFFT=OFF
   cmake --build .

Enable CUDA FFT support:

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DalpakaV_DEP_CUFFT=ON -Dalpaka_DEP_CUDA=ON -Dalpaka_CUDA_NvidiaGpu=ON
   cmake --build .

Enable HIP FFT support:

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DalpakaV_DEP_ROCFFT=ON -Dalpaka_DEP_HIP=ON -Dalpaka_HIP_AmdGpu=ON
   cmake --build .
