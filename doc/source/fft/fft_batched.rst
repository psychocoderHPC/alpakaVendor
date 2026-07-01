FFT Batched Transforms
=======================

alpakaVendor supports batched FFT transforms where multiple independent transforms are executed in a single plan.

Batch configuration
-------------------

Set the batch count and distance between batch elements:

.. code-block:: cpp

   auto plan = PlanBuilder<std::complex<float>, Extents<uint32_t, 1u>>{}
       .c2c()
       .extents({N})
       .batch(batchSize)
       .distances(inDistance, outDistance)
       .build(device);

Distance semantics
------------------

The distance specifies the number of elements between consecutive batch elements:

- **inDistance**: Elements between input batch i and batch i+1
- **outDistance**: Elements between output batch i and batch i+1

If distance is 0, it defaults to the product of the transform extents.

Batched C2C example
-------------------

.. literalinclude:: ../../../doc/code/batched_c2c.cpp
   :language: C++
   :start-after: //! [batched-c2c]
   :end-before: //! [batched-c2c]

.. raw:: html

   <details>
   <summary><strong>Full source: batched_c2c.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/batched_c2c.cpp
   :language: C++
   :caption: batched_c2c.cpp

.. raw:: html

   </details>

Batched R2C/C2R example
-----------------------

.. literalinclude:: ../../../doc/code/batched_r2c.cpp
   :language: C++
   :start-after: //! [batched-r2c]
   :end-before: //! [batched-r2c]

.. raw:: html

   <details>
   <summary><strong>Full source: batched_r2c.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/batched_r2c.cpp
   :language: C++
   :caption: batched_r2c.cpp

.. raw:: html

   </details>

Performance notes
-----------------

- Batched transforms are more efficient than individual transforms
- Contiguous batches (distance = product of extents) perform best
- cuFFT and FFTW both optimize batched execution internally

Validation
----------

- Batch must be >= 1
- Distances must not cause buffer overlaps
- Input/output distances must match the transform layout
