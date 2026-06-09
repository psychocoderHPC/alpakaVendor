/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace alpaka::fft
{
    enum class Transform
    {
        c2c,
        r2c,
        c2r
    };

    enum class Direction
    {
        forward,
        backward
    };

    enum class Normalization
    {
        none
    };

    enum class Placement
    {
        inPlace,
        outOfPlace
    };

    enum class WorkspacePolicy
    {
        backendManaged,
        userProvided
    };

    template<std::size_t T_dim>
    using Extents = std::array<std::size_t, T_dim>;

    template<std::size_t T_dim>
    using Strides = std::array<std::size_t, T_dim>;

    template<std::size_t T_dim>
    struct Layout
    {
        Extents<T_dim> extents{};
        Strides<T_dim> inStrides{};
        Strides<T_dim> outStrides{};
        std::size_t batch = 1u;
        std::size_t inDistance = 0u;
        std::size_t outDistance = 0u;
    };

    struct PlanOptions
    {
        Normalization normalization = Normalization::none;
        Placement placement = Placement::outOfPlace;
        WorkspacePolicy workspacePolicy = WorkspacePolicy::backendManaged;
    };

    template<typename T>
    concept RealScalar = std::same_as<std::remove_cv_t<T>, float> || std::same_as<std::remove_cv_t<T>, double>;

    template<typename T>
    struct is_alpaka_complex : std::false_type
    {
    };

    template<typename T>
    struct is_alpaka_complex<alpaka::math::Complex<T>> : std::bool_constant<RealScalar<T>>
    {
    };

    template<typename T>
    concept ComplexScalar = is_alpaka_complex<std::remove_cv_t<T>>::value;

    template<typename T>
    struct Complex;

    template<>
    struct Complex<float>
    {
        using type = alpaka::math::Complex<float>;
    };

    template<>
    struct Complex<double>
    {
        using type = alpaka::math::Complex<double>;
    };

    template<typename T>
    using Complex_t = typename Complex<std::remove_cv_t<T>>::type;

    template<typename T>
    struct Real;

    template<RealScalar T>
    struct Real<T>
    {
        using type = std::remove_cv_t<T>;
    };

    template<ComplexScalar T>
    struct Real<T>
    {
        using type = typename std::remove_cv_t<T>::value_type;
    };

    template<typename T>
    using Real_t = typename Real<T>::type;

    template<std::size_t T_dim>
    [[nodiscard]] constexpr std::size_t product(Extents<T_dim> const& extents)
    {
        std::size_t result = 1u;
        for(auto e : extents)
            result *= e;
        return result;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr Strides<T_dim> contiguousStrides(Extents<T_dim> const& extents)
    {
        Strides<T_dim> strides{};
        std::size_t current = 1u;
        for(std::size_t i = T_dim; i-- > 0u;)
        {
            strides[i] = current;
            current *= extents[i];
        }
        return strides;
    }

    /**
     * Return the packed complex extent for the last dimension of an R2C transform.
     *
     * Only the non-redundant Hermitian half-spectrum is stored.
     */
    [[nodiscard]] constexpr std::size_t r2cComplexExtent(std::size_t realExtent)
    {
        return realExtent / 2u + 1u;
    }

    /**
     * Recover the logical real extent from a packed C2R spectrum extent.
     *
     * This describes the transform domain size, not the padded in-place storage size.
     */
    [[nodiscard]] constexpr std::size_t c2rLogicalRealExtent(std::size_t complexExtent)
    {
        if(complexExtent == 0u)
            throw std::invalid_argument("Complex extent for C2R must be non-zero.");
        return 2u * (complexExtent - 1u);
    }

    /** Return the padded real-storage extent required for in-place C2R/R2C layouts. */
    [[nodiscard]] constexpr std::size_t c2rPaddedRealExtent(std::size_t complexExtent)
    {
        return 2u * complexExtent;
    }

    /** Return the padded real-storage extent required for in-place R2C layouts. */
    [[nodiscard]] constexpr std::size_t r2cPaddedRealExtent(std::size_t realExtent)
    {
        return 2u * r2cComplexExtent(realExtent);
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr Extents<T_dim> r2cLogicalComplexExtents(Extents<T_dim> extents)
    {
        extents[T_dim - 1u] = r2cComplexExtent(extents[T_dim - 1u]);
        return extents;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr Extents<T_dim> c2rLogicalRealExtents(Extents<T_dim> extents)
    {
        extents[T_dim - 1u] = c2rLogicalRealExtent(extents[T_dim - 1u]);
        return extents;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr Extents<T_dim> c2rPhysicalRealStorageExtents(Extents<T_dim> extents)
    {
        extents[T_dim - 1u] = c2rPaddedRealExtent(extents[T_dim - 1u]);
        return extents;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr Extents<T_dim> r2cInPlaceRealStorageExtents(Extents<T_dim> extents)
    {
        extents[T_dim - 1u] = r2cPaddedRealExtent(extents[T_dim - 1u]);
        return extents;
    }

    template<typename T_Real, std::size_t T_dim>
    requires RealScalar<T_Real>
    struct InPlaceRealStorage
    {
        Extents<T_dim> logicalRealExtents{};
        Extents<T_dim> physicalRealExtents{};
        Extents<T_dim> logicalComplexExtents{};
        std::size_t logicalRealElements = 0u;
        std::size_t physicalRealElements = 0u;
        std::size_t logicalComplexElements = 0u;
    };

    template<std::size_t T_dim>
    struct FftBufferExtents
    {
        Extents<T_dim> logicalRealExtents{};
        Extents<T_dim> physicalRealExtents{};
        Extents<T_dim> logicalComplexExtents{};
    };

    /**
     * Describe the logical and physical extents for a real buffer that may be used in-place.
     *
     * The physical real extents include the vendor-required padding in the last dimension.
     */
    template<typename T_Real, std::size_t T_dim>
    requires RealScalar<T_Real>
    [[nodiscard]] constexpr auto makeInPlaceRealStorage(Extents<T_dim> logicalRealExtents)
    {
        auto physicalRealExtents = r2cInPlaceRealStorageExtents(logicalRealExtents);
        auto logicalComplexExtents = r2cLogicalComplexExtents(logicalRealExtents);
        return InPlaceRealStorage<T_Real, T_dim>{
            .logicalRealExtents = logicalRealExtents,
            .physicalRealExtents = physicalRealExtents,
            .logicalComplexExtents = logicalComplexExtents,
            .logicalRealElements = product(logicalRealExtents),
            .physicalRealElements = product(physicalRealExtents),
            .logicalComplexElements = product(logicalComplexExtents)};
    }

    /**
     * Derive the real/complex extent views that refer to the same FFT allocation.
     *
     * For real-valued buffers this includes padded physical storage; for complex-valued buffers the input extents
     * are treated as the logical packed spectrum shape.
     */
    template<typename T_Value, std::size_t T_dim>
    [[nodiscard]] constexpr auto makeFftBufferExtents(Extents<T_dim> extents)
    {
        if constexpr(RealScalar<T_Value>)
        {
            auto storage = makeInPlaceRealStorage<T_Value>(extents);
            return FftBufferExtents<T_dim>{
                .logicalRealExtents = storage.logicalRealExtents,
                .physicalRealExtents = storage.physicalRealExtents,
                .logicalComplexExtents = storage.logicalComplexExtents};
        }
        else
        {
            return FftBufferExtents<T_dim>{
                .logicalRealExtents = c2rLogicalRealExtents(extents),
                .physicalRealExtents = c2rPhysicalRealStorageExtents(extents),
                .logicalComplexExtents = extents};
        }
    }
} // namespace alpaka::fft
