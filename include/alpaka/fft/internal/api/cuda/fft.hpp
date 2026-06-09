/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#if ALPAKAV_HAS_CUFFT
namespace alpaka::fft::internal
{
    template<typename T>
    struct CufftTraits;

    template<>
    struct CufftTraits<float>
    {
        using complex_type = cufftComplex;
        static constexpr cufftType c2c = CUFFT_C2C;
        static constexpr cufftType r2c = CUFFT_R2C;
        static constexpr cufftType c2r = CUFFT_C2R;
    };

    template<>
    struct CufftTraits<double>
    {
        using complex_type = cufftDoubleComplex;
        static constexpr cufftType c2c = CUFFT_Z2Z;
        static constexpr cufftType r2c = CUFFT_D2Z;
        static constexpr cufftType c2r = CUFFT_Z2D;
    };

    template<typename T_Value, std::size_t T_dim>
    struct PlanImpl<alpaka::api::Cuda, T_Value, T_dim>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using traits = CufftTraits<real_type>;

        Transform m_transform;
        Layout<T_dim> m_layout;
        PlanOptions m_options;
        cufftHandle m_handle = 0;
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
            , m_handle{std::exchange(other.m_handle, 0)}
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
                m_handle = std::exchange(other.m_handle, 0);
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
            if(m_handle != 0)
                cufftDestroy(m_handle);
            m_handle = 0;
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

        [[nodiscard]] cufftType type() const
        {
            if(m_transform == Transform::c2c)
                return traits::c2c;
            if(m_transform == Transform::r2c)
                return traits::r2c;
            return traits::c2r;
        }

        void check(cufftResult result, char const* what) const
        {
            if(result != CUFFT_SUCCESS)
                throw std::invalid_argument(
                    std::string{what} + " failed with cuFFT error code " + std::to_string(int(result)));
        }

        void createPlan(auto& queue)
        {
            auto dimsVals = dims();
            auto inEmbedVals = inEmbed();
            auto outEmbedVals = outEmbed();
            check(cufftCreate(&m_handle), "cufftCreate");
            check(cufftSetStream(m_handle, queue.getNativeHandle()), "cufftSetStream");
            check(
                cufftSetAutoAllocation(m_handle, m_options.workspacePolicy == WorkspacePolicy::backendManaged ? 1 : 0),
                "cufftSetAutoAllocation");
            size_t workSize = 0u;
            check(
                cufftMakePlanMany64(
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
                "cufftMakePlanMany64");
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
            validate(bytes >= m_workspaceBytes, "Provided cuFFT workspace is too small.");
            check(cufftSetWorkArea(m_handle, ptr), "cufftSetWorkArea");
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            queue.enqueueHostFn(
                [this, &queue, &in, &out, direction]()
                {
                    check(cufftSetStream(m_handle, queue.getNativeHandle()), "cufftSetStream");
                    if constexpr(ComplexScalar<T_Value>)
                    {
                        if constexpr(std::same_as<real_type, float>)
                            check(
                                cufftExecC2C(
                                    m_handle,
                                    reinterpret_cast<cufftComplex*>(removeCvPtr(in.data())),
                                    reinterpret_cast<cufftComplex*>(out.data()),
                                    direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE),
                                "cufftExecC2C");
                        else
                            check(
                                cufftExecZ2Z(
                                    m_handle,
                                    reinterpret_cast<cufftDoubleComplex*>(removeCvPtr(in.data())),
                                    reinterpret_cast<cufftDoubleComplex*>(out.data()),
                                    direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE),
                                "cufftExecZ2Z");
                    }
                    else
                    {
                        using InValue = std::remove_cv_t<std::remove_pointer_t<decltype(in.data())>>;
                        if constexpr(std::same_as<InValue, real_type>)
                        {
                            validate(direction == Direction::forward, "R2C only supports forward execution.");
                            if constexpr(std::same_as<real_type, float>)
                                check(
                                    cufftExecR2C(
                                        m_handle,
                                        removeCvPtr(in.data()),
                                        reinterpret_cast<cufftComplex*>(out.data())),
                                    "cufftExecR2C");
                            else
                                check(
                                    cufftExecD2Z(
                                        m_handle,
                                        removeCvPtr(in.data()),
                                        reinterpret_cast<cufftDoubleComplex*>(out.data())),
                                    "cufftExecD2Z");
                        }
                        else
                        {
                            validate(direction == Direction::backward, "C2R only supports backward execution.");
                            if constexpr(std::same_as<real_type, float>)
                                check(
                                    cufftExecC2R(
                                        m_handle,
                                        reinterpret_cast<cufftComplex*>(removeCvPtr(in.data())),
                                        out.data()),
                                    "cufftExecC2R");
                            else
                                check(
                                    cufftExecZ2D(
                                        m_handle,
                                        reinterpret_cast<cufftDoubleComplex*>(removeCvPtr(in.data())),
                                        out.data()),
                                    "cufftExecZ2D");
                        }
                    }
                });
            alpaka::onHost::wait(queue);
        }
    };
} // namespace alpaka::fft::internal
#endif
