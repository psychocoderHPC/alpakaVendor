/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/fft.hpp"
#include "alpaka/fft/onHost/alloc.hpp"

#include <memory>

#include "SharedBufferFFT.hpp"

namespace alpaka::fft::onHost
{
    template<
        typename T_Value,
        alpaka::concepts::Vector T_Extents = alpaka::fft::Extents<uint32_t, 1u>>
    class PlanBuilder;

    template<typename T_Api, typename T_Value, alpaka::concepts::Vector T_Extents>
    class Plan
    {
    public:
        using api_type = T_Api;
        using value_type = T_Value;
        using extents_type = T_Extents;
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;
        static constexpr uint32_t dim = T_Extents::dim();
        using impl_type = alpaka::fft::internal::PlanImpl<T_Api, T_Value, T_Extents>;

        Plan() = delete;

        Plan(auto& queue, Transform transform, Layout<T_Extents> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
            , m_impl{std::make_shared<impl_type>(queue, transform, layout, options)}
        {
        }

        Plan(Plan const&) = default;
        Plan& operator=(Plan const&) = default;
        Plan(Plan&&) noexcept = default;
        Plan& operator=(Plan&&) noexcept = default;

        [[nodiscard]] auto transform() const noexcept
        {
            return m_transform;
        }

        [[nodiscard]] auto const& layout() const noexcept
        {
            return m_layout;
        }

        [[nodiscard]] auto const& options() const noexcept
        {
            return m_options;
        }

        [[nodiscard]] auto workspaceBytes() const noexcept
        {
            return m_impl->workspaceBytes();
        }

        void setWorkspace(void* ptr, std::size_t bytes)
        {
            m_impl->setWorkspace(ptr, bytes);
        }

        template<typename T_In, typename T_Out>
        void execute(auto& queue, T_In const& in, T_Out& out, Direction direction)
        {
            m_impl->execute(queue, in, out, direction);
            m_impl->trackCompletion(queue);
        }

    private:
        Transform m_transform;
        Layout<T_Extents> m_layout;
        PlanOptions m_options;
        std::shared_ptr<impl_type> m_impl;
    };

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    class PlanBuilder
    {
    public:
        using extents_type = T_Extents;
        using index_type = alpaka::trait::GetValueType_t<T_Extents>;
        static constexpr uint32_t dim = T_Extents::dim();

        PlanBuilder& c2c()
        {
            m_transform = Transform::c2c;
            return *this;
        }

        PlanBuilder& r2c()
        {
            m_transform = Transform::r2c;
            return *this;
        }

        PlanBuilder& c2r()
        {
            m_transform = Transform::c2r;
            return *this;
        }

        PlanBuilder& extents(alpaka::concepts::VectorOrScalar auto const& value)
        {
            m_layout.extents = alpaka::fft::internal::normalizeVectorOrScalar<T_Extents>(value);
            m_layout.inStrides = {};
            m_layout.outStrides = {};
            m_layout.inDistance = static_cast<index_type>(0u);
            m_layout.outDistance = static_cast<index_type>(0u);
            return *this;
        }

        PlanBuilder& batch(index_type value)
        {
            m_layout.batch = value;
            return *this;
        }

        PlanBuilder& strides(alpaka::concepts::Vector auto const& in, alpaka::concepts::Vector auto const& out)
        {
            m_layout.inStrides = alpaka::fft::internal::castVec<T_Extents>(in);
            m_layout.outStrides = alpaka::fft::internal::castVec<T_Extents>(out);
            return *this;
        }

        PlanBuilder& distances(index_type inDistance, index_type outDistance)
        {
            m_layout.inDistance = inDistance;
            m_layout.outDistance = outDistance;
            return *this;
        }

        PlanBuilder& inPlace()
        {
            m_options.placement = Placement::inPlace;
            return *this;
        }

        PlanBuilder& outOfPlace()
        {
            m_options.placement = Placement::outOfPlace;
            return *this;
        }

        PlanBuilder& backendManagedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::backendManaged;
            return *this;
        }

        PlanBuilder& userProvidedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::userProvided;
            return *this;
        }

        [[nodiscard]] auto build(auto& queue) const
        {
            using Api = decltype(alpaka::getApi(queue));
            return Plan<Api, T_Value, T_Extents>{queue, m_transform, m_layout, m_options};
        }

    private:
        Transform m_transform = Transform::c2c;
        Layout<T_Extents> m_layout{};
        PlanOptions m_options{};
    };

    template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
    void executeForward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out)
    {
        plan.execute(queue, in, out, Direction::forward);
    }

    template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
    void executeBackward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out)
    {
        plan.execute(queue, in, out, Direction::backward);
    }

    template<
        typename T_Api,
        RealScalar T_Real,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment,
        typename T_Plan,
        typename T_Queue>
    auto executeR2CInPlace(
        T_Queue& queue,
        T_Plan& plan,
        SharedBufferFFT<T_Api, T_Real, T_Extents, T_MemAlignment>& buffer)
    {
        auto complexBuffer = buffer.asComplex();
        plan.execute(queue, buffer, complexBuffer, Direction::forward);
        return complexBuffer;
    }

    template<
        typename T_Api,
        ComplexScalar T_Complex,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment,
        typename T_Plan,
        typename T_Queue>
    auto executeC2RInPlace(
        T_Queue& queue,
        T_Plan& plan,
        SharedBufferFFT<T_Api, T_Complex, T_Extents, T_MemAlignment>& buffer)
    {
        auto realBuffer = buffer.asReal();
        plan.execute(queue, buffer, realBuffer, Direction::backward);
        return realBuffer;
    }
} // namespace alpaka::fft::onHost
