FFT Tutorial
=============

This tutorial walks through the core concepts of alpakaVendor's FFT module step by step.

Core concepts
-------------

1. **Plan**: An FFT plan describes the transform type, dimensions, and options.
2. **Layout**: Defines the transform extents, strides, and batch configuration.
3. **Buffer**: FFT data is stored in alpaka buffers, wrapped by ``SharedBufferFFT`` for in-place transforms.
4. **Queue**: Execution happens on alpaka queues that represent execution streams.

Memory layout convention
++++++++++++++++++++++++

alpakaVendor follows alpaka's memory layout convention: **the last index is the fast-moving index** (row-major order).

For a 2D M x N transform:

.. code-block:: cpp

   // M rows, N columns
   // Index 0 (M): slow-moving (row)
   // Index 1 (N): fast-moving (column)
   auto extents = alpaka::Vec<uint32_t, 2u>{M, N};

   // Linear index: row * N + col
   data[row * N + col] = value;

For a 3D D x H x W transform:

.. code-block:: cpp

   // Index 0 (D): slowest-moving (depth)
   // Index 1 (H): middle (height)
   // Index 2 (W): fast-moving (width)
   auto extents = alpaka::Vec<uint32_t, 3u>{D, H, W};

   // Linear index: (depth * H + row) * W + col
   data[(depth * H + row) * W + col] = value;

When copying your own data into alpaka buffers, ensure the fast-moving index corresponds to the contiguous dimension.

Step 1: Select a backend and create a queue
-------------------------------------------

.. literalinclude:: ../../../doc/code/tutorial_full.cpp
   :language: C++
   :start-after: //! [tutorial-setup]
   :end-before: //! [tutorial-setup]

Step 2: 1D C2C out-of-place transform
-------------------------------------

.. literalinclude:: ../../../doc/code/tutorial_full.cpp
   :language: C++
   :start-after: //! [tutorial-c2c]
   :end-before: //! [tutorial-c2c]

Step 3: 2D C2C out-of-place transform
-------------------------------------

.. literalinclude:: ../../../doc/code/tutorial_full.cpp
   :language: C++
   :start-after: //! [tutorial-2d]
   :end-before: //! [tutorial-2d]

Step 4: 1D R2C/C2R in-place roundtrip
-------------------------------------

.. literalinclude:: ../../../doc/code/tutorial_full.cpp
   :language: C++
   :start-after: //! [tutorial-inplace]
   :end-before: //! [tutorial-inplace]

Complete example
----------------

.. raw:: html

   <details>
   <summary><strong>Full source: tutorial_full.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/tutorial_full.cpp
   :language: C++
   :caption: tutorial_full.cpp

.. raw:: html

   </details>
