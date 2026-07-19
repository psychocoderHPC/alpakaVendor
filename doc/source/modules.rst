Modules
=======

alpakaVendor is organized by functional modules. Each module gets its own documentation subtree so new domains can be added without mixing beginner material, module guides, and API notes.

Current and planned modules
---------------------------

FFT
   Available today. Covers complex and real transforms, backend selection, batched execution, and in-place storage helpers.

BLAS
   Available. Covers Level-1 vector routines, GEMV, GEMM, strided batched GEMM, and triangular solve.

Parallel primitives
   Planned. Intended for data-parallel building blocks such as elementwise transforms, sorting, scans, and reductions.

Documentation layout convention
--------------------------------

The documentation is split into three layers:

- ``basic/`` for project-wide introduction and installation
- one top-level directory per module such as ``fft/``
- shared generated reference pages such as Doxygen

This keeps the structure stable as more modules are added later.
