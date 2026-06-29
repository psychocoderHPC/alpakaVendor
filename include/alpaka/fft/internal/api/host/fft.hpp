/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#if ALPAKAV_HAS_FFTW
namespace alpaka::fft::internal
{
    template<typename T>
    struct FftwTraits;

    template<>
    struct FftwTraits<float>
    {
        using plan_type = fftwf_plan;
        using complex_type = fftwf_complex;
        static constexpr unsigned flags = FFTW_ESTIMATE;
    };

    template<>
    struct FftwTraits<double>
    {
        using plan_type = fftw_plan;
        using complex_type = fftw_complex;
        static constexpr unsigned flags = FFTW_ESTIMATE;
    };

    template<typename T_Value, std::size_t T_dim>
    struct PlanImpl<alpaka::api::Host, T_Value, T_dim>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using traits = FftwTraits<real_type>;
        using plan_type = typename traits::plan_type;

        Transform m_transform;
        Layout<T_dim> m_layout;
        PlanOptions m_options;
        plan_type m_plan = nullptr;

        PlanImpl(auto&, Transform transform, Layout<T_dim> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
        {
            validateConfig();
            createPlan();
        }

        PlanImpl(PlanImpl&& other) noexcept
            : m_transform{other.m_transform}
            , m_layout{other.m_layout}
            , m_options{other.m_options}
            , m_plan{std::exchange(other.m_plan, nullptr)}
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
                m_plan = std::exchange(other.m_plan, nullptr);
            }
            return *this;
        }

        ~PlanImpl()
        {
            destroy();
        }

        void destroy() noexcept
        {
            if constexpr(std::same_as<real_type, float>)
            {
                if(m_plan != nullptr)
                    fftwf_destroy_plan(m_plan);
            }
            else
            {
                if(m_plan != nullptr)
                    fftw_destroy_plan(m_plan);
            }
            m_plan = nullptr;
        }

        void validateConfig() const
        {
            validate(T_dim >= 1u && T_dim <= 3u, "FFT only supports dimensions 1..3.");
            validate(m_layout.batch >= 1u, "FFT batch must be >= 1.");
            for(auto e : m_layout.extents)
                validate(e > 0u, "FFT extents must be non-zero.");
            validate(m_options.normalization == Normalization::none, "Only Normalization::none is supported.");
            if(m_options.workspacePolicy == WorkspacePolicy::userProvided)
                throw std::invalid_argument("FFTW backend does not support user-provided workspace in v1.");
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

        [[nodiscard]] auto n() const
        {
            std::array<int, T_dim> nVals{};
            for(std::size_t i = 0; i < T_dim; ++i)
                nVals[i] = static_cast<int>(m_layout.extents[i]);
            return nVals;
        }

        [[nodiscard]] auto inEmbed() const
        {
            auto ext = expectedInputExtents(m_layout, m_transform, m_options.placement);
            std::array<int, T_dim> vals{};
            for(std::size_t i = 0; i < T_dim; ++i)
                vals[i] = static_cast<int>(ext[i]);
            return vals;
        }

        [[nodiscard]] auto outEmbed() const
        {
            auto ext = expectedOutputExtents(m_layout, m_transform, m_options.placement);
            std::array<int, T_dim> vals{};
            for(std::size_t i = 0; i < T_dim; ++i)
                vals[i] = static_cast<int>(ext[i]);
            return vals;
        }

        void createPlan()
        {
            auto nVals = n();
            auto inEmbedVals = inEmbed();
            auto outEmbedVals = outEmbed();
            auto inDistance = static_cast<int>(expectedInDistance(m_layout, m_transform, m_options.placement));
            auto outDistance = static_cast<int>(expectedOutDistance(m_layout, m_transform, m_options.placement));
            int batch = static_cast<int>(m_layout.batch);
            if constexpr(std::same_as<real_type, float>)
            {
                if(m_transform == Transform::c2c)
                {
                    m_plan = fftwf_plan_many_dft(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        nullptr,
                        inEmbedVals.data(),
                        1,
                        inDistance,
                        nullptr,
                        outEmbedVals.data(),
                        1,
                        outDistance,
                        FFTW_FORWARD,
                        traits::flags);
                }
                else if(m_transform == Transform::r2c)
                {
                    m_plan = fftwf_plan_many_dft_r2c(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        nullptr,
                        inEmbedVals.data(),
                        1,
                        inDistance,
                        nullptr,
                        outEmbedVals.data(),
                        1,
                        outDistance,
                        traits::flags);
                }
                else
                {
                    m_plan = fftwf_plan_many_dft_c2r(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        nullptr,
                        inEmbedVals.data(),
                        1,
                        inDistance,
                        nullptr,
                        outEmbedVals.data(),
                        1,
                        outDistance,
                        traits::flags);
                }
            }
            else
            {
                if(m_transform == Transform::c2c)
                {
                    m_plan = fftw_plan_many_dft(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        nullptr,
                        inEmbedVals.data(),
                        1,
                        inDistance,
                        nullptr,
                        outEmbedVals.data(),
                        1,
                        outDistance,
                        FFTW_FORWARD,
                        traits::flags);
                }
                else if(m_transform == Transform::r2c)
                {
                    m_plan = fftw_plan_many_dft_r2c(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        nullptr,
                        inEmbedVals.data(),
                        1,
                        inDistance,
                        nullptr,
                        outEmbedVals.data(),
                        1,
                        outDistance,
                        traits::flags);
                }
                else
                {
                    m_plan = fftw_plan_many_dft_c2r(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        nullptr,
                        inEmbedVals.data(),
                        1,
                        inDistance,
                        nullptr,
                        outEmbedVals.data(),
                        1,
                        outDistance,
                        traits::flags);
                }
            }
            validate(m_plan != nullptr, "FFTW plan creation failed.");
        }

        [[nodiscard]] std::size_t workspaceBytes() const noexcept
        {
            return 0u;
        }

        void setWorkspace(void*, std::size_t)
        {
            throw std::invalid_argument("FFTW backend does not support user workspace.");
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            queue.enqueueHostFn(
                [&in, &out, direction, this]()
                {
                    auto nVals = n();
                    auto inEmbedVals = inEmbed();
                    auto outEmbedVals = outEmbed();
                    auto inDistance = static_cast<int>(expectedInDistance(m_layout, m_transform, m_options.placement));
                    auto outDistance
                        = static_cast<int>(expectedOutDistance(m_layout, m_transform, m_options.placement));
                    int batch = static_cast<int>(m_layout.batch);

                    if constexpr(ComplexScalar<T_Value>)
                    {
                        validate(
                            direction == Direction::forward || direction == Direction::backward,
                            "Invalid direction.");
                        if(m_layout.batch > 1u)
                        {
                            validateBatchedViewExtents<decltype(in), T_dim>(
                                in,
                                m_layout.extents,
                                m_layout.batch,
                                "Input");
                            validateBatchedViewExtents<decltype(out), T_dim>(
                                out,
                                m_layout.extents,
                                m_layout.batch,
                                "Output");
                        }
                        else
                        {
                            validateViewExtents<decltype(in), T_dim>(in, m_layout.extents, "Input");
                            validateViewExtents<decltype(out), T_dim>(out, m_layout.extents, "Output");
                        }
                        auto* inPtr = reinterpret_cast<typename traits::complex_type*>(removeCvPtr(in.data()));
                        auto* outPtr = reinterpret_cast<typename traits::complex_type*>(out.data());
                        if constexpr(std::same_as<real_type, float>)
                        {
                            auto plan = fftwf_plan_many_dft(
                                static_cast<int>(T_dim),
                                nVals.data(),
                                batch,
                                inPtr,
                                inEmbedVals.data(),
                                1,
                                inDistance,
                                outPtr,
                                outEmbedVals.data(),
                                1,
                                outDistance,
                                direction == Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD,
                                traits::flags);
                            validate(plan != nullptr, "FFTW plan creation failed.");
                            fftwf_execute_dft(plan, inPtr, outPtr);
                            fftwf_destroy_plan(plan);
                        }
                        else
                        {
                            auto plan = fftw_plan_many_dft(
                                static_cast<int>(T_dim),
                                nVals.data(),
                                batch,
                                inPtr,
                                inEmbedVals.data(),
                                1,
                                inDistance,
                                outPtr,
                                outEmbedVals.data(),
                                1,
                                outDistance,
                                direction == Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD,
                                traits::flags);
                            validate(plan != nullptr, "FFTW plan creation failed.");
                            fftw_execute_dft(plan, inPtr, outPtr);
                            fftw_destroy_plan(plan);
                        }
                    }
                    else
                    {
                        using InValue = std::remove_cv_t<std::remove_pointer_t<decltype(in.data())>>;
                        if constexpr(std::same_as<InValue, real_type>)
                        {
                            validate(direction == Direction::forward, "R2C only supports forward execution.");
                            if(m_layout.batch > 1u)
                            {
                                validateBatchedViewExtents<decltype(in), T_dim>(
                                    in,
                                    expectedInputExtents(m_layout, Transform::r2c, m_options.placement),
                                    m_layout.batch,
                                    "Input");
                                validateBatchedViewExtents<decltype(out), T_dim>(
                                    out,
                                    expectedOutputExtents(m_layout, Transform::r2c, m_options.placement),
                                    m_layout.batch,
                                    "Output");
                            }
                            else
                            {
                                validateViewExtents<decltype(in), T_dim>(
                                    in,
                                    expectedInputExtents(m_layout, Transform::r2c, m_options.placement),
                                    "Input");
                                validateViewExtents<decltype(out), T_dim>(
                                    out,
                                    expectedOutputExtents(m_layout, Transform::r2c, m_options.placement),
                                    "Output");
                            }
                            if constexpr(std::same_as<real_type, float>)
                            {
                                auto plan = fftwf_plan_many_dft_r2c(
                                    static_cast<int>(T_dim),
                                    nVals.data(),
                                    batch,
                                    removeCvPtr(in.data()),
                                    inEmbedVals.data(),
                                    1,
                                    inDistance,
                                    reinterpret_cast<typename traits::complex_type*>(out.data()),
                                    outEmbedVals.data(),
                                    1,
                                    outDistance,
                                    traits::flags);
                                validate(plan != nullptr, "FFTW plan creation failed.");
                                fftwf_execute_dft_r2c(
                                    plan,
                                    removeCvPtr(in.data()),
                                    reinterpret_cast<typename traits::complex_type*>(out.data()));
                                fftwf_destroy_plan(plan);
                            }
                            else
                            {
                                auto plan = fftw_plan_many_dft_r2c(
                                    static_cast<int>(T_dim),
                                    nVals.data(),
                                    batch,
                                    removeCvPtr(in.data()),
                                    inEmbedVals.data(),
                                    1,
                                    inDistance,
                                    reinterpret_cast<typename traits::complex_type*>(out.data()),
                                    outEmbedVals.data(),
                                    1,
                                    outDistance,
                                    traits::flags);
                                validate(plan != nullptr, "FFTW plan creation failed.");
                                fftw_execute_dft_r2c(
                                    plan,
                                    removeCvPtr(in.data()),
                                    reinterpret_cast<typename traits::complex_type*>(out.data()));
                                fftw_destroy_plan(plan);
                            }
                        }
                        else
                        {
                            validate(direction == Direction::backward, "C2R only supports backward execution.");
                            if(m_layout.batch > 1u)
                            {
                                validateBatchedViewExtents<decltype(in), T_dim>(
                                    in,
                                    expectedInputExtents(m_layout, Transform::c2r, m_options.placement),
                                    m_layout.batch,
                                    "Input");
                                validateBatchedViewExtents<decltype(out), T_dim>(
                                    out,
                                    expectedOutputExtents(m_layout, Transform::c2r, m_options.placement),
                                    m_layout.batch,
                                    "Output");
                            }
                            else
                            {
                                validateViewExtents<decltype(in), T_dim>(
                                    in,
                                    expectedInputExtents(m_layout, Transform::c2r, m_options.placement),
                                    "Input");
                                validateViewExtents<decltype(out), T_dim>(
                                    out,
                                    expectedOutputExtents(m_layout, Transform::c2r, m_options.placement),
                                    "Output");
                            }
                            if constexpr(std::same_as<real_type, float>)
                            {
                                auto plan = fftwf_plan_many_dft_c2r(
                                    static_cast<int>(T_dim),
                                    nVals.data(),
                                    batch,
                                    reinterpret_cast<typename traits::complex_type*>(removeCvPtr(in.data())),
                                    inEmbedVals.data(),
                                    1,
                                    inDistance,
                                    out.data(),
                                    outEmbedVals.data(),
                                    1,
                                    outDistance,
                                    traits::flags);
                                validate(plan != nullptr, "FFTW plan creation failed.");
                                fftwf_execute_dft_c2r(
                                    plan,
                                    reinterpret_cast<typename traits::complex_type*>(removeCvPtr(in.data())),
                                    out.data());
                                fftwf_destroy_plan(plan);
                            }
                            else
                            {
                                auto plan = fftw_plan_many_dft_c2r(
                                    static_cast<int>(T_dim),
                                    nVals.data(),
                                    batch,
                                    reinterpret_cast<typename traits::complex_type*>(removeCvPtr(in.data())),
                                    inEmbedVals.data(),
                                    1,
                                    inDistance,
                                    out.data(),
                                    outEmbedVals.data(),
                                    1,
                                    outDistance,
                                    traits::flags);
                                validate(plan != nullptr, "FFTW plan creation failed.");
                                fftw_execute_dft_c2r(
                                    plan,
                                    reinterpret_cast<typename traits::complex_type*>(removeCvPtr(in.data())),
                                    out.data());
                                fftw_destroy_plan(plan);
                            }
                        }
                    }
                });
            alpaka::onHost::wait(queue);
        }
    };
} // namespace alpaka::fft::internal
#endif
