/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#if ALPAKAV_HAS_ONEMKL
namespace alpaka::fft::internal
{
    template<typename T>
    struct OneMklTraits;

    template<>
    struct OneMklTraits<float>
    {
        static constexpr oneapi::mkl::dft::precision precision = oneapi::mkl::dft::precision::SINGLE;
    };

    template<>
    struct OneMklTraits<double>
    {
        static constexpr oneapi::mkl::dft::precision precision = oneapi::mkl::dft::precision::DOUBLE;
    };

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    struct PlanImpl<alpaka::api::OneApi, T_Value, T_Extents>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using complex_type = std::complex<real_type>;
        using traits = OneMklTraits<real_type>;
        static constexpr auto domain
            = ComplexScalar<T_Value> ? oneapi::mkl::dft::domain::COMPLEX : oneapi::mkl::dft::domain::REAL;
        using descriptor_type = oneapi::mkl::dft::descriptor<traits::precision, domain>;

        Transform m_transform;
        static constexpr uint32_t T_dim = T_Extents::dim();
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;

        Layout<T_Extents> m_layout;
        PlanOptions m_options;
        std::unique_ptr<descriptor_type> m_descriptor;
        sycl::queue m_syclQueue;
        std::size_t m_workspaceBytes = 0u;
        bool m_workspaceConfigured = false;

        PlanImpl(auto& queue, Transform transform, Layout<T_Extents> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
        {
            static_assert(
                sizeof(alpaka::math::Complex<real_type>) == sizeof(complex_type),
                "alpaka::math::Complex must match std::complex storage for oneMKL interop.");
            static_assert(
                alignof(alpaka::math::Complex<real_type>) == alignof(complex_type),
                "alpaka::math::Complex must match std::complex alignment for oneMKL interop.");

            validateConfig();
            createDescriptor(queue);
        }

        PlanImpl(PlanImpl&& other) noexcept
            : m_transform{other.m_transform}
            , m_layout{other.m_layout}
            , m_options{other.m_options}
            , m_descriptor{std::move(other.m_descriptor)}
            , m_syclQueue{std::move(other.m_syclQueue)}
            , m_workspaceBytes{std::exchange(other.m_workspaceBytes, 0u)}
            , m_workspaceConfigured{std::exchange(other.m_workspaceConfigured, false)}
        {
            std::lock_guard lock(other.m_pendingWaitsMutex);
            m_pendingWaits = std::move(other.m_pendingWaits);
        }

        PlanImpl& operator=(PlanImpl&& other) noexcept
        {
            if(this != &other)
            {
                waitForPending();
                m_transform = other.m_transform;
                m_layout = other.m_layout;
                m_options = other.m_options;
                m_descriptor = std::move(other.m_descriptor);
                m_syclQueue = std::move(other.m_syclQueue);
                m_workspaceBytes = std::exchange(other.m_workspaceBytes, 0u);
                m_workspaceConfigured = std::exchange(other.m_workspaceConfigured, false);
                {
                    std::scoped_lock lock(m_pendingWaitsMutex, other.m_pendingWaitsMutex);
                    m_pendingWaits = std::move(other.m_pendingWaits);
                }
            }
            return *this;
        }

        ~PlanImpl()
        {
            waitForPending();
        }

        void validateConfig() const
        {
            validate(T_dim >= 1u && T_dim <= 3u, "FFT only supports dimensions 1..3.");
            validate(m_layout.batch >= 1u, "FFT batch must be >= 1.");
            for(uint32_t i = 0u; i < T_dim; ++i)
                validate(m_layout.extents[i] > static_cast<index_type>(0u), "FFT extents must be non-zero.");
            validate(m_options.normalization == Normalization::none, "Only Normalization::none is supported.");
            if constexpr(ComplexScalar<T_Value>)
                validate(m_transform == Transform::c2c, "Complex plan value type only supports C2C.");
            else
                validate(m_transform != Transform::c2c, "Real plan value type only supports R2C/C2R.");
            if(!areZero(m_layout.inStrides))
                validate(
                    m_layout.inStrides == expectedInStrides(m_layout, m_transform, m_options.placement),
                    "Only contiguous input layout is supported.");
            if(!areZero(m_layout.outStrides))
                validate(
                    m_layout.outStrides == expectedOutStrides(m_layout, m_transform, m_options.placement),
                    "Only contiguous output layout is supported.");
        }

        [[nodiscard]] auto lengths() const
        {
            std::vector<std::int64_t> result(T_dim);
            for(uint32_t i = 0u; i < T_dim; ++i)
                result[i] = static_cast<std::int64_t>(m_layout.extents[i]);
            return result;
        }

        template<alpaka::concepts::Vector T_Vec>
        [[nodiscard]] static auto strideVector(T_Vec const& vec)
        {
            std::vector<std::int64_t> result(T_dim + 1u, 0);
            for(uint32_t i = 0u; i < T_dim; ++i)
                result[i + 1u] = static_cast<std::int64_t>(vec[i]);
            return result;
        }

        [[nodiscard]] auto forwardDomainExtents() const
        {
            if constexpr(ComplexScalar<T_Value>)
                return expectedInputExtents(m_layout, m_transform, m_options.placement);
            else if(m_options.placement == Placement::inPlace)
                return r2cPaddedRealExtent(m_layout.extents);
            else
                return m_layout.extents;
        }

        [[nodiscard]] auto backwardDomainExtents() const
        {
            if constexpr(ComplexScalar<T_Value>)
                return expectedOutputExtents(m_layout, m_transform, m_options.placement);
            else
                return r2cComplexExtent(m_layout.extents);
        }

        [[nodiscard]] auto forwardDomainDistance() const
        {
            if(m_layout.inDistance != static_cast<index_type>(0u))
                return m_layout.inDistance;
            return product(forwardDomainExtents());
        }

        [[nodiscard]] auto backwardDomainDistance() const
        {
            if(m_layout.outDistance != static_cast<index_type>(0u))
                return m_layout.outDistance;
            return product(backwardDomainExtents());
        }

        void createDescriptor(auto& queue)
        {
            auto device = queue.getDevice();
            auto [syclDevice, syclContext] = device.getNativeHandle();
            m_syclQueue = sycl::queue(syclContext, syclDevice, {sycl::property::queue::in_order{}});
            m_descriptor = std::make_unique<descriptor_type>(lengths());

            if(m_options.placement == Placement::outOfPlace)
            {
                m_descriptor->set_value(
                    oneapi::mkl::dft::config_param::PLACEMENT,
                    oneapi::mkl::dft::config_value::NOT_INPLACE);
            }

            auto const forwardStrides = strideVector(contiguousStrides(forwardDomainExtents()));
            auto const backwardStrides = strideVector(contiguousStrides(backwardDomainExtents()));
            m_descriptor->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, forwardStrides);
            m_descriptor->set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, backwardStrides);
            m_descriptor->set_value(
                oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                static_cast<std::int64_t>(m_layout.batch));
            m_descriptor->set_value(
                oneapi::mkl::dft::config_param::FWD_DISTANCE,
                static_cast<std::int64_t>(forwardDomainDistance()));
            m_descriptor->set_value(
                oneapi::mkl::dft::config_param::BWD_DISTANCE,
                static_cast<std::int64_t>(backwardDomainDistance()));

            if(m_options.workspacePolicy == WorkspacePolicy::userProvided)
            {
                m_descriptor->set_value(
                    oneapi::mkl::dft::config_param::WORKSPACE,
                    oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
            }

            m_descriptor->commit(m_syclQueue);

            try
            {
                std::int64_t workspaceBytes = 0;
                m_descriptor->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES, &workspaceBytes);
                m_workspaceBytes = static_cast<std::size_t>(workspaceBytes);
            }
            catch(...)
            {
                m_workspaceBytes = 0u;
            }
        }

        [[nodiscard]] std::size_t workspaceBytes() const noexcept
        {
            return m_workspaceBytes;
        }

        void setWorkspace(void* ptr, std::size_t bytes)
        {
            validate(
                m_options.workspacePolicy == WorkspacePolicy::userProvided,
                "Plan does not use user-provided workspace.");
            validate(bytes >= m_workspaceBytes, "Provided oneMKL workspace is too small.");
            m_descriptor->set_workspace(reinterpret_cast<std::uint8_t*>(ptr));
            m_workspaceConfigured = true;
        }

        void trackCompletion(auto&)
        {
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            if(m_options.workspacePolicy == WorkspacePolicy::userProvided && m_workspaceBytes > 0u)
                validate(m_workspaceConfigured, "User-provided oneMKL workspace must be bound before execution.");

            alpaka::onHost::wait(queue);

            auto* rawInPtr = removeCvPtr(in.data());
            auto* rawOutPtr = removeCvPtr(out.data());
            sycl::event event;

            if constexpr(ComplexScalar<T_Value>)
            {
                auto* inPtr = reinterpret_cast<complex_type*>(rawInPtr);
                auto* outPtr = reinterpret_cast<complex_type*>(rawOutPtr);
                event = direction == Direction::forward
                            ? oneapi::mkl::dft::compute_forward(*m_descriptor, inPtr, outPtr)
                            : oneapi::mkl::dft::compute_backward(*m_descriptor, inPtr, outPtr);
            }
            else
            {
                using InValue = std::remove_cv_t<std::remove_pointer_t<decltype(rawInPtr)>>;
                if constexpr(std::same_as<InValue, real_type>)
                {
                    validate(direction == Direction::forward, "R2C only supports forward execution.");
                    event = oneapi::mkl::dft::compute_forward(
                        *m_descriptor,
                        rawInPtr,
                        reinterpret_cast<complex_type*>(rawOutPtr));
                }
                else
                {
                    validate(direction == Direction::backward, "C2R only supports backward execution.");
                    event = oneapi::mkl::dft::compute_backward(
                        *m_descriptor,
                        reinterpret_cast<complex_type*>(rawInPtr),
                        rawOutPtr);
                }
            }

            auto sharedEvent = std::make_shared<sycl::event>(std::move(event));
            queue.enqueueHostFn([sharedEvent]() { sharedEvent->wait_and_throw(); });
            std::lock_guard lock(m_pendingWaitsMutex);
            m_pendingWaits.emplace_back([sharedEvent]() { sharedEvent->wait_and_throw(); });
        }

    private:
        void waitForPending() noexcept
        {
            std::vector<std::function<void()>> waits;
            {
                std::lock_guard lock(m_pendingWaitsMutex);
                waits.swap(m_pendingWaits);
            }
            for(auto& waitFn : waits)
            {
                try
                {
                    waitFn();
                }
                catch(...)
                {
                }
            }
        }

        std::mutex m_pendingWaitsMutex;
        std::vector<std::function<void()>> m_pendingWaits;
    };
} // namespace alpaka::fft::internal
#endif
