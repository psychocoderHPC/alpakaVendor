FFT In-place Real Transforms
=============================

In-place R2C/C2R transforms require special handling because the real and complex representations have different storage layouts.

Storage layout
--------------

For a 1D transform of size N:

- **Logical real extent**: N
- **Physical real storage**: 2 * (N/2 + 1) (padded)
- **Logical complex extent**: N/2 + 1

For a 2D transform of size M x N:

- **Logical real extent**: M x N
- **Physical real storage**: M x (2 * (N/2 + 1)) (padded)
- **Logical complex extent**: M x (N/2 + 1)

Padding helpers
---------------

.. literalinclude:: ../../../doc/code/inplace_helpers.cpp
   :language: C++
   :start-after: //! [inplace-helpers]
   :end-before: //! [inplace-helpers]

.. raw:: html

   <details>
   <summary><strong>Full source: inplace_helpers.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/inplace_helpers.cpp
   :language: C++
   :caption: inplace_helpers.cpp

.. raw:: html

   </details>

SharedBufferFFT
---------------

``SharedBufferFFT`` manages the dual real/complex view of a single allocation:

.. literalinclude:: ../../../doc/code/inplace_shared_buffer.cpp
   :language: C++
   :start-after: //! [inplace-shared-buffer]
   :end-before: //! [inplace-shared-buffer]

.. raw:: html

   <details>
   <summary><strong>Full source: inplace_shared_buffer.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/inplace_shared_buffer.cpp
   :language: C++
   :caption: inplace_shared_buffer.cpp

.. raw:: html

   </details>

Complete in-place roundtrip
---------------------------

.. literalinclude:: ../../../doc/code/inplace_roundtrip.cpp
   :language: C++
   :start-after: //! [inplace-roundtrip]
   :end-before: //! [inplace-roundtrip]

.. raw:: html

   <details>
   <summary><strong>Full source: inplace_roundtrip.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/inplace_roundtrip.cpp
   :language: C++
   :caption: inplace_roundtrip.cpp

.. raw:: html

   </details>

Semantic rules
--------------

After an in-place R2C transform, the complex view is the valid interpretation.
After an in-place C2R transform, the real view is the valid interpretation.

.. warning::

   Accessing the wrong view after a transform leads to undefined behavior.
   The library enforces this through ``SharedBufferFFT`` which tracks the active view.

Numerical behavior
------------------

With ``Normalization::none``:

- Forward R2C followed by backward C2R produces the original data multiplied by N.
- The scaling factor is the logical transform size, not the physical storage size.
