/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#if ALPAKAV_HAS_HIPFFT
namespace alpaka::fft::internal
{
    template<typename T>
    struct HipfftTraits;

    template<>
    struct HipfftTraits<float>
    {
        using complex_type = hipfftComplex;
        static constexpr hipfftType c2c = HIPFFT_C2C;
        static constexpr hipfftType r2c = HIPFFT_R2C;
        static constexpr hipfftType c2r = HIPFFT_C2R;
    };

    template<>
    struct HipfftTraits<double>
    {
        using complex_type = hipfftDoubleComplex;
        static constexpr hipfftType c2c = HIPFFT_Z2Z;
        static constexpr hipfftType r2c = HIPFFT_D2Z;
        static constexpr hipfftType c2r = HIPFFT_Z2D;
    };

    template<typename T_Value, std::size_t T_dim>
    struct PlanImpl<alpaka::api::Hip, T_Value, T_dim>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using traits = HipfftTraits<real_type>;

        Transform m_transform;
        Layout<T_dim> m_layout;
        PlanOptions m_options;
        hipfftHandle m_handle = nullptr;
        std::size_t m_workspaceBytes = 0u;

        PlanImpl(auto& queue, Transform transform, Layout<T_dim> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
        {
            validateConfig();
            createPlan(queue);
        }

        PlanImpl(PlanImpl&& other) noexcept
            : m_transform{other.m_transform}
            , m_layout{other.m_layout}
            , m_options{other.m_options}
            , m_handle{std::exchange(other.m_handle, nullptr)}
            , m_workspaceBytes{std::exchange(other.m_workspaceBytes, 0u)}
        {
        }

        PlanImpl& operator=(PlanImpl&& other) noexcept
        {
            if(this != &other)
            {
                destroy();
                m_transform = other.m_transform;
                m_layout = other.m_layout;
                m_options = other.m_options;
                m_handle = std::exchange(other.m_handle, nullptr);
                m_workspaceBytes = std::exchange(other.m_workspaceBytes, 0u);
            }
            return *this;
        }

        ~PlanImpl()
        {
            destroy();
        }

        void destroy() noexcept
        {
            if(m_handle != nullptr)
                hipfftDestroy(m_handle);
            m_handle = nullptr;
        }

        void validateConfig() const
        {
            validate(T_dim >= 1u && T_dim <= 3u, "FFT only supports dimensions 1..3.");
            validate(m_layout.batch >= 1u, "FFT batch must be >= 1.");
            for(auto e : m_layout.extents)
                validate(e > 0u, "FFT extents must be non-zero.");
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

        [[nodiscard]] auto dims() const
        {
            std::array<long long, T_dim> result{};
            for(std::size_t i = 0; i < T_dim; ++i)
                result[i] = static_cast<long long>(m_layout.extents[i]);
            return result;
        }

        [[nodiscard]] auto inEmbed() const
        {
            auto ext = expectedInputExtents(m_layout, m_transform, m_options.placement);
            std::array<long long, T_dim> result{};
            for(std::size_t i = 0; i < T_dim; ++i)
                result[i] = static_cast<long long>(ext[i]);
            return result;
        }

        [[nodiscard]] auto outEmbed() const
        {
            auto ext = expectedOutputExtents(m_layout, m_transform, m_options.placement);
            std::array<long long, T_dim> result{};
            for(std::size_t i = 0; i < T_dim; ++i)
                result[i] = static_cast<long long>(ext[i]);
            return result;
        }

        [[nodiscard]] hipfftType type() const
        {
            if(m_transform == Transform::c2c)
                return traits::c2c;
            if(m_transform == Transform::r2c)
                return traits::r2c;
            return traits::c2r;
        }

        void check(hipfftResult result, char const* what) const
        {
            if(result != HIPFFT_SUCCESS)
                throw std::invalid_argument(
                    std::string{what} + " failed with hipFFT error code " + std::to_string(int(result)));
        }

        void createPlan(auto& queue)
        {
            auto dimsVals = dims();
            auto inEmbedVals = inEmbed();
            auto outEmbedVals = outEmbed();
            check(hipfftCreate(&m_handle), "hipfftCreate");
            check(hipfftSetStream(m_handle, queue.getNativeHandle()), "hipfftSetStream");
            check(
                hipfftSetAutoAllocation(
                    m_handle,
                    m_options.workspacePolicy == WorkspacePolicy::backendManaged ? 1 : 0),
                "hipfftSetAutoAllocation");
            size_t workSize = 0u;
            check(
                hipfftMakePlanMany64(
                    m_handle,
                    static_cast<int>(T_dim),
                    dimsVals.data(),
                    inEmbedVals.data(),
                    1,
                    static_cast<long long>(expectedInDistance(m_layout, m_transform, m_options.placement)),
                    outEmbedVals.data(),
                    1,
                    static_cast<long long>(expectedOutDistance(m_layout, m_transform, m_options.placement)),
                    type(),
                    static_cast<long long>(m_layout.batch),
                    &workSize),
                "hipfftMakePlanMany64");
            m_workspaceBytes = workSize;
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
            validate(bytes >= m_workspaceBytes, "Provided hipFFT workspace is too small.");
            check(hipfftSetWorkArea(m_handle, ptr), "hipfftSetWorkArea");
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            queue.enqueueHostFn(
                [this, &queue, &in, &out, direction]()
                {
                    check(hipfftSetStream(m_handle, queue.getNativeHandle()), "hipfftSetStream");
                    if constexpr(ComplexScalar<T_Value>)
                    {
                        if constexpr(std::same_as<real_type, float>)
                            check(
                                hipfftExecC2C(
                                    m_handle,
                                    reinterpret_cast<hipfftComplex*>(removeCvPtr(in.data())),
                                    reinterpret_cast<hipfftComplex*>(out.data()),
                                    direction == Direction::forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD),
                                "hipfftExecC2C");
                        else
                            check(
                                hipfftExecZ2Z(
                                    m_handle,
                                    reinterpret_cast<hipfftDoubleComplex*>(removeCvPtr(in.data())),
                                    reinterpret_cast<hipfftDoubleComplex*>(out.data()),
                                    direction == Direction::forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD),
                                "hipfftExecZ2Z");
                    }
                    else
                    {
                        using InValue = std::remove_cv_t<std::remove_pointer_t<decltype(in.data())>>;
                        if constexpr(std::same_as<InValue, real_type>)
                        {
                            validate(direction == Direction::forward, "R2C only supports forward execution.");
                            if constexpr(std::same_as<real_type, float>)
                                check(
                                    hipfftExecR2C(
                                        m_handle,
                                        removeCvPtr(in.data()),
                                        reinterpret_cast<hipfftComplex*>(out.data())),
                                    "hipfftExecR2C");
                            else
                                check(
                                    hipfftExecD2Z(
                                        m_handle,
                                        removeCvPtr(in.data()),
                                        reinterpret_cast<hipfftDoubleComplex*>(out.data())),
                                    "hipfftExecD2Z");
                        }
                        else
                        {
                            validate(direction == Direction::backward, "C2R only supports backward execution.");
                            if constexpr(std::same_as<real_type, float>)
                                check(
                                    hipfftExecC2R(
                                        m_handle,
                                        reinterpret_cast<hipfftComplex*>(removeCvPtr(in.data())),
                                        out.data()),
                                    "hipfftExecC2R");
                            else
                                check(
                                    hipfftExecZ2D(
                                        m_handle,
                                        reinterpret_cast<hipfftDoubleComplex*>(removeCvPtr(in.data())),
                                        out.data()),
                                    "hipfftExecZ2D");
                        }
                    }
                });
            alpaka::onHost::wait(queue);
        }
    };
} // namespace alpaka::fft::internal
#endif
