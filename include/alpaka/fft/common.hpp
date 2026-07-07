/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
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

    template<typename T_Index, uint32_t T_dim>
    using Extents = alpaka::Vec<T_Index, T_dim>;

    template<typename T_Index, uint32_t T_dim>
    using Strides = alpaka::Vec<T_Index, T_dim>;

    /** Memory layout descriptor for an FFT plan.
     *
     * Strides and distances are expressed in **bytes** (not elements), matching the alpaka pitch convention
     * returned by `getPitches()`. When a backend library requires element counts, the conversion is performed
     * automatically at execution time.
     */
    template<alpaka::concepts::Vector T_Extents>
    struct Layout
    {
        using extents_type = T_Extents;
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;
        using byte_type = std::size_t;
        static constexpr uint32_t dim = T_Extents::dim();

        T_Extents extents{};
        alpaka::Vec<byte_type, dim> inStrides{}; ///< Input byte-strides per dimension (alpaka pitch order).
        alpaka::Vec<byte_type, dim> outStrides{}; ///< Output byte-strides per dimension (alpaka pitch order).
        index_type batch = static_cast<index_type>(1u);
        byte_type inDistance = 0u; ///< Byte distance between consecutive input batches.
        byte_type outDistance = 0u; ///< Byte distance between consecutive output batches.
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

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto contiguousStrides(T_Extents const& extents)
    {
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;
        T_Extents strides{};
        index_type current = static_cast<index_type>(1u);
        for(uint32_t i = T_Extents::dim(); i-- > 0u;)
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
    template<std::integral T_Index>
    [[nodiscard]] constexpr T_Index r2cComplexExtent(T_Index realExtent)
    {
        return realExtent / static_cast<T_Index>(2u) + static_cast<T_Index>(1u);
    }

    template<alpaka::concepts::Vector T_Extents>
    requires std::integral<alpaka::trait::GetValueType_t<T_Extents>>
    [[nodiscard]] constexpr T_Extents r2cComplexExtent(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = r2cComplexExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    /**
     * Recover the logical real extent from a packed C2R spectrum extent.
     *
     * This describes the transform domain size, not the padded in-place storage size.
     */
    template<std::integral T_Index>
    [[nodiscard]] constexpr T_Index c2rLogicalRealExtent(T_Index complexExtent)
    {
        if(complexExtent == static_cast<T_Index>(0u))
            throw std::invalid_argument("Complex extent for C2R must be non-zero.");
        return static_cast<T_Index>(2u) * (complexExtent - static_cast<T_Index>(1u));
    }

    template<alpaka::concepts::Vector T_Extents>
    requires std::integral<alpaka::trait::GetValueType_t<T_Extents>>
    [[nodiscard]] constexpr T_Extents c2rLogicalRealExtent(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = c2rLogicalRealExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    /** Return the padded real-storage extent required for in-place C2R/R2C layouts. */
    template<std::integral T_Index>
    [[nodiscard]] constexpr T_Index c2rPaddedRealExtent(T_Index complexExtent)
    {
        return static_cast<T_Index>(2u) * complexExtent;
    }

    template<alpaka::concepts::Vector T_Extents>
    requires std::integral<alpaka::trait::GetValueType_t<T_Extents>>
    [[nodiscard]] constexpr T_Extents c2rPaddedRealExtent(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = c2rPaddedRealExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    /** Return the padded real-storage extent required for in-place R2C layouts. */
    template<std::integral T_Index>
    [[nodiscard]] constexpr T_Index r2cPaddedRealExtent(T_Index realExtent)
    {
        return static_cast<T_Index>(2u) * r2cComplexExtent(realExtent);
    }

    template<alpaka::concepts::Vector T_Extents>
    requires std::integral<alpaka::trait::GetValueType_t<T_Extents>>
    [[nodiscard]] constexpr T_Extents r2cPaddedRealExtent(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = r2cPaddedRealExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    /**
     * Describe the logical and physical extents for a real buffer that may be used in-place.
     *
     * The physical real extents include the vendor-required padding in the last dimension.
     */
    template<typename T_Real, alpaka::concepts::Vector T_Extents>
    requires RealScalar<T_Real>
    struct InPlaceRealStorage
    {
        using extents_type = T_Extents;
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;

        T_Extents logicalRealExtents{};
        T_Extents physicalRealExtents{};
        T_Extents logicalComplexExtents{};
        index_type logicalRealElements = static_cast<index_type>(0u);
        index_type physicalRealElements = static_cast<index_type>(0u);
        index_type logicalComplexElements = static_cast<index_type>(0u);
    };

    template<alpaka::concepts::Vector T_Extents>
    struct FftBufferExtents
    {
        T_Extents logicalRealExtents{};
        T_Extents physicalRealExtents{};
        T_Extents logicalComplexExtents{};
    };

    template<typename T_Real>
    requires RealScalar<T_Real>
    [[nodiscard]] constexpr auto makeInPlaceRealStorage(
        alpaka::concepts::VectorOrScalar auto const& logicalRealExtents)
    {
        auto extents = [&]()
        {
            if constexpr(alpaka::concepts::Vector<std::remove_cvref_t<decltype(logicalRealExtents)>>)
                return std::remove_cvref_t<decltype(logicalRealExtents)>{logicalRealExtents};
            else
                return alpaka::Vec<std::remove_cvref_t<decltype(logicalRealExtents)>, 1u>::fill(logicalRealExtents);
        }();
        auto physicalRealExtents = r2cPaddedRealExtent(extents);
        auto logicalComplexExtents = r2cComplexExtent(extents);
        return InPlaceRealStorage<T_Real, decltype(extents)>{
            .logicalRealExtents = extents,
            .physicalRealExtents = physicalRealExtents,
            .logicalComplexExtents = logicalComplexExtents,
            .logicalRealElements = extents.product(),
            .physicalRealElements = physicalRealExtents.product(),
            .logicalComplexElements = logicalComplexExtents.product()};
    }

    /**
     * Derive the real/complex extent views that refer to the same FFT allocation.
     *
     * For real-valued buffers this includes padded physical storage; for complex-valued buffers the input extents
     * are treated as the logical packed spectrum shape.
     */
    template<typename T_Value>
    [[nodiscard]] constexpr auto makeFftBufferExtents(alpaka::concepts::VectorOrScalar auto const& extentsArg)
    {
        auto extents = [&]()
        {
            if constexpr(alpaka::concepts::Vector<std::remove_cvref_t<decltype(extentsArg)>>)
                return std::remove_cvref_t<decltype(extentsArg)>{extentsArg};
            else
                return alpaka::Vec<std::remove_cvref_t<decltype(extentsArg)>, 1u>::fill(extentsArg);
        }();
        if constexpr(RealScalar<T_Value>)
        {
            auto storage = makeInPlaceRealStorage<T_Value>(extents);
            return FftBufferExtents<decltype(extents)>{
                .logicalRealExtents = storage.logicalRealExtents,
                .physicalRealExtents = storage.physicalRealExtents,
                .logicalComplexExtents = storage.logicalComplexExtents};
        }
        else
        {
            return FftBufferExtents<decltype(extents)>{
                .logicalRealExtents = c2rLogicalRealExtent(extents),
                .physicalRealExtents = c2rPaddedRealExtent(extents),
                .logicalComplexExtents = extents};
        }
    }
} // namespace alpaka::fft
