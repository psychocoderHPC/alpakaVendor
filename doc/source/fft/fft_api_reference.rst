FFT API Reference
==================

This page documents the public FFT API of alpakaVendor.

Common types
------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       enum class Transform { c2c, r2c, c2r };
       enum class Direction { forward, backward };
       enum class Normalization { none };
       enum class Placement { inPlace, outOfPlace };
       enum class WorkspacePolicy { backendManaged, userProvided };

       template<std::size_t T_dim>
       using Extents = std::array<std::size_t, T_dim>;

       template<std::size_t T_dim>
       using Strides = std::array<std::size_t, T_dim>;
   }

Transform types
+++++++++++++++

- ``Transform::c2c``: Complex-to-complex transform
- ``Transform::r2c``: Real-to-complex transform
- ``Transform::c2r``: Complex-to-real transform

Placement
+++++++++

- ``Placement::outOfPlace``: Input and output in separate buffers (default)
- ``Placement::inPlace``: Input and output share the same buffer

Workspace policy
++++++++++++++++

- ``WorkspacePolicy::backendManaged``: Backend allocates workspace automatically (default)
- ``WorkspacePolicy::userProvided``: User provides workspace via ``setWorkspace()``

Layout
------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<std::size_t T_dim>
       struct Layout
       {
           Extents<T_dim> extents{};      // Logical FFT extents
           Strides<T_dim> inStrides{};    // Element strides (not byte strides)
           Strides<T_dim> outStrides{};   // Element strides (not byte strides)
           std::size_t batch = 1u;         // Number of transforms in batch
           std::size_t inDistance = 0u;    // Distance between batch elements (input)
           std::size_t outDistance = 0u;   // Distance between batch elements (output)
       };
   }

Plan options
------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       struct PlanOptions
       {
           Normalization normalization = Normalization::none;
           Placement placement = Placement::outOfPlace;
           WorkspacePolicy workspacePolicy = WorkspacePolicy::backendManaged;
       };
   }

Type traits
-----------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<typename T>
       concept RealScalar = std::same_as<T, float> || std::same_as<T, double>;

       template<typename T>
       concept ComplexScalar = /* std::complex<RealScalar> */;

       template<typename T>
       using Complex_t = typename Complex<T>::type;  // float -> std::complex<float>

       template<typename T>
       using Real_t = typename Real<T>::type;  // std::complex<float> -> float
   }

Padding helpers
---------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       // Complex extent for R2C: N/2 + 1
       [[nodiscard]] constexpr std::size_t r2cComplexExtent(std::size_t realExtent);

       // Padded real extent for in-place R2C: 2 * (N/2 + 1)
       [[nodiscard]] constexpr std::size_t r2cPaddedRealExtent(std::size_t realExtent);

       // Logical complex extents for R2C transform
       template<std::size_t T_dim>
       [[nodiscard]] constexpr Extents<T_dim> r2cLogicalComplexExtents(Extents<T_dim> realExtents);

       // Physical real storage extents for in-place R2C
       template<std::size_t T_dim>
       [[nodiscard]] constexpr Extents<T_dim> r2cInPlaceRealStorageExtents(Extents<T_dim> realExtents);

       // Product of all extents
       template<std::size_t T_dim>
       [[nodiscard]] constexpr std::size_t product(Extents<T_dim> const& extents);

       // Contiguous strides for given extents
       template<std::size_t T_dim>
       [[nodiscard]] constexpr Strides<T_dim> contiguousStrides(Extents<T_dim> const& extents);
   }

In-place real storage
---------------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<typename T_Real, std::size_t T_dim>
       struct InPlaceRealStorage
       {
           Extents<T_dim> logicalRealExtents{};
           Extents<T_dim> physicalRealExtents{};
           Extents<T_dim> logicalComplexExtents{};
           std::size_t logicalRealElements = 0u;
           std::size_t physicalRealElements = 0u;
           std::size_t logicalComplexElements = 0u;
       };

       template<typename T_Real, std::size_t T_dim>
       [[nodiscard]] constexpr auto makeInPlaceRealStorage(Extents<T_dim> logicalRealExtents);
   }

PlanBuilder
-----------

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       template<typename T_Value, std::size_t T_dim>
       class PlanBuilder
       {
       public:
           PlanBuilder& c2c();
           PlanBuilder& r2c();
           PlanBuilder& c2r();
           PlanBuilder& extents(Extents<T_dim> value);
           PlanBuilder& batch(std::size_t value);
           PlanBuilder& strides(Strides<T_dim> in, Strides<T_dim> out);
           PlanBuilder& distances(std::size_t inDistance, std::size_t outDistance);
           PlanBuilder& inPlace();
           PlanBuilder& outOfPlace();
           PlanBuilder& backendManagedWorkspace();
           PlanBuilder& userProvidedWorkspace();

           [[nodiscard]] auto build(auto& queue) const;
       };
   }

Execution helpers
-----------------

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       // Execute forward/out-of-place transform
       template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
       void executeForward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out);

       // Execute backward/out-of-place transform
       template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
       void executeBackward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out);

       // Execute in-place R2C transform
       template<typename T_Api, RealScalar T_Real, ...>
       auto executeR2CInPlace(T_Queue& queue, T_Plan& plan, SharedBufferFFT<T_Api, T_Real, ...>& buffer);

       // Execute in-place C2R transform
       template<typename T_Api, ComplexScalar T_Complex, ...>
       auto executeC2RInPlace(
           T_Queue& queue, T_Plan& plan,
           SharedBufferFFT<T_Api, T_Complex, ...>& buffer);
   }

Allocation helpers
------------------

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       // Allocate FFT-managed buffer with automatic real/complex reinterpretation support
       template<typename T_Type>
       [[nodiscard]] auto allocForFFT(auto const& device, auto const& extents);

       // Allocate unified memory buffer for FFT
       template<typename T_Type>
       [[nodiscard]] auto allocUnifiedForFFT(auto const& device, auto const& extents);

       // Allocate mapped memory buffer for FFT
       template<typename T_Type>
       [[nodiscard]] auto allocMappedForFFT(auto const& device, auto const& extents);

       // Allocate deferred buffer for FFT
       template<typename T_Type>
       [[nodiscard]] auto allocDeferredForFFT(auto const& queue, auto const& extents);
   }

SharedBufferFFT
---------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<alpaka::concepts::Api T_Api, typename T_Type, ...>
       struct SharedBufferFFT : alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>
       {
           // Get byte capacity of the buffer
           [[nodiscard]] auto byteCapacity() const noexcept;

           // Get reference count
           [[nodiscard]] auto getUseCount() const noexcept;

           // Check if buffer is valid
           [[nodiscard]] explicit operator bool() const noexcept;

           // Reinterpret buffer as different type using explicit extents
           template<typename T_Other>
           [[nodiscard]] auto reinterpretBuffer(auto const& extents) const;

           // FFT-aware convenience views
           [[nodiscard]] auto asReal() const;
           [[nodiscard]] auto asComplex() const;
       };
   }
