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

    template<alpaka::concepts::Vector T_Extents>
    struct Layout
    {
        using extents_type = T_Extents;
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;
        static constexpr uint32_t dim = T_Extents::dim();

        T_Extents extents{};
        T_Extents inStrides{};
        T_Extents outStrides{};
        index_type batch = static_cast<index_type>(1u);
        index_type inDistance = static_cast<index_type>(0u);
        index_type outDistance = static_cast<index_type>(0u);
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

    template<typename T_From, typename T_To>
    inline constexpr bool isLosslessIntegralUpcastV = std::integral<T_From> && std::integral<T_To>
        && (std::same_as<std::remove_cv_t<T_From>, std::remove_cv_t<T_To>>
            || ((std::is_signed_v<T_From> == std::is_signed_v<T_To>)
                && (std::numeric_limits<T_To>::digits >= std::numeric_limits<T_From>::digits))
            || (std::is_unsigned_v<T_From> && std::is_signed_v<T_To>
                && (std::numeric_limits<T_To>::digits > std::numeric_limits<T_From>::digits)));

    template<typename T_Value, uint32_t T_dim>
    [[nodiscard]] constexpr auto filledVec(T_Value value)
    {
        alpaka::Vec<T_Value, T_dim> result{};
        for(uint32_t i = 0u; i < T_dim; ++i)
            result[i] = value;
        return result;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto product(T_Extents const& extents)
    {
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;
        index_type result = static_cast<index_type>(1u);
        for(uint32_t i = 0u; i < T_Extents::dim(); ++i)
            result *= extents[i];
        return result;
    }

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

    template<typename T_Index>
    requires std::integral<std::remove_cv_t<T_Index>>
    [[nodiscard]] constexpr T_Index r2cComplexExtent(T_Index realExtent)
    {
        return realExtent / static_cast<T_Index>(2u) + static_cast<T_Index>(1u);
    }

    template<typename T_Index>
    requires std::integral<std::remove_cv_t<T_Index>>
    [[nodiscard]] constexpr T_Index c2rLogicalRealExtent(T_Index complexExtent)
    {
        if(complexExtent == static_cast<T_Index>(0u))
            throw std::invalid_argument("Complex extent for C2R must be non-zero.");
        return static_cast<T_Index>(2u) * (complexExtent - static_cast<T_Index>(1u));
    }

    template<typename T_Index>
    requires std::integral<std::remove_cv_t<T_Index>>
    [[nodiscard]] constexpr T_Index c2rPaddedRealExtent(T_Index complexExtent)
    {
        return static_cast<T_Index>(2u) * complexExtent;
    }

    template<typename T_Index>
    requires std::integral<std::remove_cv_t<T_Index>>
    [[nodiscard]] constexpr T_Index r2cPaddedRealExtent(T_Index realExtent)
    {
        return static_cast<T_Index>(2u) * r2cComplexExtent(realExtent);
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr T_Extents r2cLogicalComplexExtents(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = r2cComplexExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr T_Extents c2rLogicalRealExtents(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = c2rLogicalRealExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr T_Extents c2rPhysicalRealStorageExtents(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = c2rPaddedRealExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr T_Extents r2cInPlaceRealStorageExtents(T_Extents extents)
    {
        extents[T_Extents::dim() - 1u] = r2cPaddedRealExtent(extents[T_Extents::dim() - 1u]);
        return extents;
    }

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
    [[nodiscard]] constexpr auto makeInPlaceRealStorage(alpaka::concepts::VectorOrScalar auto const& logicalRealExtents)
    {
        auto extents = [&]()
        {
            if constexpr(alpaka::concepts::Vector<std::remove_cvref_t<decltype(logicalRealExtents)>>)
                return std::remove_cvref_t<decltype(logicalRealExtents)>{logicalRealExtents};
            else
                return filledVec<std::remove_cvref_t<decltype(logicalRealExtents)>, 1u>(logicalRealExtents);
        }();
        auto physicalRealExtents = r2cInPlaceRealStorageExtents(extents);
        auto logicalComplexExtents = r2cLogicalComplexExtents(extents);
        return InPlaceRealStorage<T_Real, decltype(extents)>{
            .logicalRealExtents = extents,
            .physicalRealExtents = physicalRealExtents,
            .logicalComplexExtents = logicalComplexExtents,
            .logicalRealElements = product(extents),
            .physicalRealElements = product(physicalRealExtents),
            .logicalComplexElements = product(logicalComplexExtents)};
    }

    template<typename T_Value>
    [[nodiscard]] constexpr auto makeFftBufferExtents(alpaka::concepts::VectorOrScalar auto const& extentsArg)
    {
        auto extents = [&]()
        {
            if constexpr(alpaka::concepts::Vector<std::remove_cvref_t<decltype(extentsArg)>>)
                return std::remove_cvref_t<decltype(extentsArg)>{extentsArg};
            else
                return filledVec<std::remove_cvref_t<decltype(extentsArg)>, 1u>(extentsArg);
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
                .logicalRealExtents = c2rLogicalRealExtents(extents),
                .physicalRealExtents = c2rPhysicalRealStorageExtents(extents),
                .logicalComplexExtents = extents};
        }
    }
} // namespace alpaka::fft
