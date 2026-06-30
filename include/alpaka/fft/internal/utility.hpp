/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/common.hpp"

namespace alpaka::fft::internal
{
    template<alpaka::concepts::Vector T_TargetVec, alpaka::concepts::Vector T_SourceVec>
    [[nodiscard]] constexpr auto castVec(T_SourceVec const& vec)
    {
        static_assert(T_TargetVec::dim() == T_SourceVec::dim(), "Extent dimensions must match.");
        using source_index_type = alpaka::trait::GetValueType_t<T_SourceVec>;
        using target_index_type = alpaka::trait::GetValueType_t<T_TargetVec>;
        static_assert(
            isLosslessIntegralUpcastV<source_index_type, target_index_type>,
            "Extent vector element types must match or upcast without precision loss.");

        T_TargetVec result{};
        for(uint32_t i = 0u; i < T_TargetVec::dim(); ++i)
            result[i] = static_cast<target_index_type>(vec[i]);
        return result;
    }

    template<alpaka::concepts::Vector T_TargetVec>
    [[nodiscard]] constexpr auto normalizeVectorOrScalar(alpaka::concepts::VectorOrScalar auto const& value)
    {
        if constexpr(alpaka::concepts::Vector<std::remove_cvref_t<decltype(value)>>)
            return castVec<T_TargetVec>(value);
        else
        {
            using source_index_type = std::remove_cvref_t<decltype(value)>;
            using target_index_type = alpaka::trait::GetValueType_t<T_TargetVec>;
            static_assert(
                isLosslessIntegralUpcastV<source_index_type, target_index_type>,
                "Extent scalar type must upcast to the extent vector element type without precision loss.");
            return filledVec<target_index_type, T_TargetVec::dim()>(static_cast<target_index_type>(value));
        }
    }

    [[nodiscard]] constexpr auto asExtentVec(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        if constexpr(alpaka::concepts::Vector<std::remove_cvref_t<decltype(extents)>>)
            return std::remove_cvref_t<decltype(extents)>{extents};
        else
            return filledVec<std::remove_cvref_t<decltype(extents)>, 1u>(extents);
    }

    template<alpaka::concepts::Vector T_Strides>
    [[nodiscard]] constexpr bool areZero(T_Strides const& strides)
    {
        for(uint32_t i = 0u; i < T_Strides::dim(); ++i)
            if(strides[i] != static_cast<alpaka::trait::GetValueType_t<T_Strides>>(0u))
                return false;
        return true;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedInputExtents(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(placement == Placement::inPlace && transform == Transform::r2c)
            return r2cPaddedRealExtent(layout.extents);
        if(transform == Transform::c2r)
            return r2cComplexExtent(layout.extents);
        return layout.extents;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedOutputExtents(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(placement == Placement::inPlace && transform == Transform::r2c)
            return r2cComplexExtent(layout.extents);
        if(transform == Transform::r2c)
            return r2cComplexExtent(layout.extents);
        return layout.extents;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedInDistance(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(layout.inDistance != static_cast<alpaka::trait::GetValueType_t<T_Extents>>(0u))
            return layout.inDistance;
        return product(expectedInputExtents(layout, transform, placement));
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedOutDistance(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(layout.outDistance != static_cast<alpaka::trait::GetValueType_t<T_Extents>>(0u))
            return layout.outDistance;
        return product(expectedOutputExtents(layout, transform, placement));
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedInStrides(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        return contiguousStrides(expectedInputExtents(layout, transform, placement));
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedOutStrides(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        return contiguousStrides(expectedOutputExtents(layout, transform, placement));
    }

    template<typename T>
    [[nodiscard]] constexpr auto removeCvPtr(T* ptr)
    {
        return const_cast<std::remove_const_t<T>*>(ptr);
    }

    template<typename T_View, alpaka::concepts::Vector T_Extents>
    void validateViewExtents(T_View const& view, T_Extents const& expected, std::string const& what)
    {
        auto const actual = castVec<T_Extents>(view.getExtents());
        if(actual != expected)
            throw std::invalid_argument(what + " extents do not match plan.");
    }

    template<typename T_View, alpaka::concepts::Vector T_Extents>
    void validateBatchedViewExtents(
        T_View const& view,
        T_Extents const& perTransformExtents,
        alpaka::trait::GetValueType_t<T_Extents> batch,
        std::string const& what)
    {
        auto const actual = castVec<T_Extents>(view.getExtents());
        if constexpr(T_Extents::dim() > 1u)
        {
            for(uint32_t i = 1u; i < T_Extents::dim(); ++i)
            {
                if(actual[i] != perTransformExtents[i])
                    throw std::invalid_argument(what + " extents do not match plan.");
            }
        }
        if(actual[0] < batch * perTransformExtents[0])
            throw std::invalid_argument(what + " first dimension too small for batched transform.");
    }

    inline void validate(bool cond, std::string const& msg)
    {
        if(!cond)
            throw std::invalid_argument(msg);
    }
} // namespace alpaka::fft::internal
