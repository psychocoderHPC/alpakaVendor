/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include "SharedBufferFFT.hpp"

namespace alpaka::fft::onHost::internal
{
    template<typename T_Type, typename T_Buffer>
    [[nodiscard]] auto wrapBuffer(T_Buffer&& buffer, auto const& fftExtents)
    {
        using Buffer = std::decay_t<T_Buffer>;
        using Api = decltype(alpaka::getApi(std::declval<Buffer>()));
        using ExtentsVec = decltype(buffer.getExtents());
        auto rawOwner = std::make_shared<Buffer>(ALPAKA_FORWARD(buffer));
        auto managedDeleter = std::make_shared<alpaka::onHost::internal::ManagedDealloc>([owner = rawOwner]() mutable
                                                                                         { owner.reset(); });
        auto metadata = std::make_shared<FftBufferMetadata<ExtentsVec>>();
        metadata->extents = fftExtents;
        std::size_t bytes = sizeof(T_Type);
        for(std::size_t i = 0; i < ExtentsVec::dim(); ++i)
            bytes *= static_cast<std::size_t>(rawOwner->getExtents()[i]);
        return SharedBufferFFT<Api, T_Type, ExtentsVec, alpaka::Alignment<>>{
            Api{},
            const_cast<T_Type*>(rawOwner->data()),
            rawOwner->getExtents(),
            rawOwner->getPitches(),
            std::move(managedDeleter),
            std::move(metadata),
            bytes,
            alpaka::Alignment<>{}};
    }

    template<typename T_Type>
    [[nodiscard]] auto fftAllocationExtents(auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::toExtents<std::decay_t<decltype(extents)>::dim()>(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        if constexpr(RealScalar<T_Type>)
            return alpaka::fft::internal::toVec(storage.physicalRealExtents);
        else
            return alpaka::fft::internal::toVec(storage.logicalComplexExtents);
    }
} // namespace alpaka::fft::onHost::internal

namespace alpaka::fft::onHost
{
    /**
     * Allocate an FFT-managed buffer.
     *
     * Real-valued allocations reserve padded physical storage so the returned buffer can be reinterpreted between
     * real and complex FFT views. Complex-valued allocations keep their logical complex extents and can be
     * reinterpreted back to their matching real FFT view.
     *
     * For manual, non-FFT-managed storage use `alpaka::onHost::alloc()` directly.
     */
    template<typename T_Type>
    [[nodiscard]] auto allocForFFT(alpaka::onHost::internal::concepts::Device auto const& device, auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::toExtents<std::decay_t<decltype(extents)>::dim()>(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(extents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::alloc<T_Type>(device, fftExtents), storage);
    }

    /**
     * Allocate a unified FFT-managed buffer.
     *
     * Real-valued allocations reserve padded physical storage so the returned buffer can be reinterpreted between
     * real and complex FFT views. Complex-valued allocations keep their logical complex extents and can be
     * reinterpreted back to their matching real FFT view.
     *
     * For manual, non-FFT-managed storage use `alpaka::onHost::allocUnified()` directly.
     */
    template<typename T_Type>
    [[nodiscard]] auto allocUnifiedForFFT(
        alpaka::onHost::internal::concepts::Device auto const& device,
        auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::toExtents<std::decay_t<decltype(extents)>::dim()>(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(extents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::allocUnified<T_Type>(device, fftExtents), storage);
    }

    /**
     * Allocate a mapped FFT-managed buffer.
     *
     * The returned buffer stores FFT reinterpretation metadata and may be converted between matching real and complex
     * FFT views.
     */
    template<typename T_Type>
    [[nodiscard]] auto allocMappedForFFT(
        alpaka::onHost::internal::concepts::Device auto const& device,
        auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::toExtents<std::decay_t<decltype(extents)>::dim()>(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(extents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::allocMapped<T_Type>(device, fftExtents), storage);
    }

    /**
     * Allocate a deferred FFT-managed buffer from a queue.
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    [[nodiscard]] auto allocDeferredForFFT(
        alpaka::onHost::Queue<T_Device, T_QueueKind> const& queue,
        auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::toExtents<std::decay_t<decltype(extents)>::dim()>(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(extents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::allocDeferred<T_Type>(queue, fftExtents), storage);
    }

    template<typename T_Type, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    [[nodiscard]] auto allocDeferedForFFT(
        alpaka::onHost::Queue<T_Device, T_QueueKind> const& queue,
        auto const& extents)
    {
        return allocDeferredForFFT<T_Type>(queue, extents);
    }
} // namespace alpaka::fft::onHost
