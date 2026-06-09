/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/common.hpp"

namespace alpaka::fft::internal
{
    template<std::size_t T_dim>
    using Vec = alpaka::Vec<std::size_t, T_dim>;

    template<std::size_t T_dim>
    [[nodiscard]] constexpr auto toVec(Extents<T_dim> const& extents)
    {
        Vec<T_dim> result{};
        for(std::size_t i = 0; i < T_dim; ++i)
            result[i] = extents[i];
        return result;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr auto toExtents(auto const& vec)
    {
        Extents<T_dim> result{};
        for(std::size_t i = 0; i < T_dim; ++i)
            result[i] = static_cast<std::size_t>(vec[i]);
        return result;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr bool areZero(Strides<T_dim> const& strides)
    {
        for(auto v : strides)
            if(v != 0u)
                return false;
        return true;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr auto expectedInputExtents(
        Layout<T_dim> const& layout,
        Transform transform,
        Placement placement)
    {
        if(placement == Placement::inPlace && transform == Transform::r2c)
            return r2cInPlaceRealStorageExtents(layout.extents);
        if(transform == Transform::c2r)
            return r2cLogicalComplexExtents(layout.extents);
        return layout.extents;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr auto expectedOutputExtents(
        Layout<T_dim> const& layout,
        Transform transform,
        Placement placement)
    {
        if(placement == Placement::inPlace && transform == Transform::r2c)
            return r2cLogicalComplexExtents(layout.extents);
        if(transform == Transform::r2c)
            return r2cLogicalComplexExtents(layout.extents);
        return layout.extents;
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr std::size_t expectedInDistance(
        Layout<T_dim> const& layout,
        Transform transform,
        Placement placement)
    {
        if(layout.inDistance != 0u)
            return layout.inDistance;
        return product(expectedInputExtents(layout, transform, placement));
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr std::size_t expectedOutDistance(
        Layout<T_dim> const& layout,
        Transform transform,
        Placement placement)
    {
        if(layout.outDistance != 0u)
            return layout.outDistance;
        return product(expectedOutputExtents(layout, transform, placement));
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr auto expectedInStrides(
        Layout<T_dim> const& layout,
        Transform transform,
        Placement placement)
    {
        return contiguousStrides(expectedInputExtents(layout, transform, placement));
    }

    template<std::size_t T_dim>
    [[nodiscard]] constexpr auto expectedOutStrides(
        Layout<T_dim> const& layout,
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

    template<typename T_View, std::size_t T_dim>
    void validateViewExtents(T_View const& view, Extents<T_dim> const& expected, std::string const& what)
    {
        auto const actual = toExtents<T_dim>(view.getExtents());
        if(actual != expected)
            throw std::invalid_argument(what + " extents do not match plan.");
    }

    template<typename T_View, std::size_t T_dim>
    void validateBatchedViewExtents(
        T_View const& view,
        Extents<T_dim> const& perTransformExtents,
        std::size_t batch,
        std::string const& what)
    {
        auto const actual = toExtents<T_dim>(view.getExtents());
        // For batched transforms, the first dimension should be batch * perTransformExtents[0]
        // and other dimensions should match
        if constexpr(T_dim > 1u)
        {
            for(std::size_t i = 1u; i < T_dim; ++i)
            {
                if(actual[i] != perTransformExtents[i])
                    throw std::invalid_argument(what + " extents do not match plan.");
            }
        }
        // First dimension should be at least batch * perTransformExtents[0]
        if(actual[0] < batch * perTransformExtents[0])
            throw std::invalid_argument(what + " first dimension too small for batched transform.");
    }

    inline void validate(bool cond, std::string const& msg)
    {
        if(!cond)
            throw std::invalid_argument(msg);
    }
} // namespace alpaka::fft::internal
