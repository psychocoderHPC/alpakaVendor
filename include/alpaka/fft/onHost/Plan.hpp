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
    template<typename T_Value, alpaka::concepts::Vector T_Extents = alpaka::fft::Extents<uint32_t, 1u>>
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

        /** Bind user-managed workspace to this plan.
         *
         * The storage must remain valid for every later execution that uses this plan. Rebinding replaces the
         * previously configured workspace for subsequent launches.
         */
        void setWorkspace(void* ptr, std::size_t bytes)
        {
            m_impl->setWorkspace(ptr, bytes);
        }

        /** Execute the transform without implicit normalization.
         *
         * Forward and backward transforms use the raw vendor-library convention. A round-trip therefore usually
         * needs an explicit division by the logical FFT size to recover the original values.
         */
        template<typename T_In, typename T_Out>
        void execute(auto& queue, T_In const& in, T_Out& out, Direction direction)
        {
            m_impl->execute(queue, in, out, direction);
            m_impl->trackCompletion(queue);
        }

        /** Keep the plan alive until the queue reaches this point.
         *
         * This mirrors alpaka's buffer `keepAlive()` helper and is useful when work using the plan has already been
         * enqueued but the last host-side plan handle is about to leave scope.
         */
        void keepAlive(auto& queue) const
        {
            queue.enqueueHostFnDeferred([impl = m_impl] {});
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

        /** Construct a plan builder with the given transform extents.
         *
         * Extents are required and must be non-zero. All other parameters have sensible defaults and can be
         * configured through the builder methods.
         */
        PlanBuilder(alpaka::concepts::VectorOrScalar auto const& extents)
            : m_layout{.extents = alpaka::fft::internal::normalizeVectorOrScalar<T_Extents>(extents)}
        {
        }

        /** Select a complex-to-complex transform. */
        PlanBuilder& c2c()
        {
            m_transform = Transform::c2c;
            return *this;
        }

        /** Select a real-to-complex forward transform. */
        PlanBuilder& r2c()
        {
            m_transform = Transform::r2c;
            return *this;
        }

        /** Select a complex-to-real backward transform. */
        PlanBuilder& c2r()
        {
            m_transform = Transform::c2r;
            return *this;
        }

        /** Set the number of transforms stored in the input/output buffers.
         *
         * Values below `1` are invalid and rejected during `build()`.
         */
        PlanBuilder& batch(index_type value)
        {
            m_layout.batch = value;
            return *this;
        }

        /** Override the element strides passed to the backend.
         *
         * Strides are expressed in elements, not bytes, and follow alpaka's layout convention where the last
         * index is the fast-moving one. Call order relative to `extents()` does not matter.
         */
        PlanBuilder& strides(alpaka::concepts::Vector auto const& in, alpaka::concepts::Vector auto const& out)
        {
            m_layout.inStrides = alpaka::fft::internal::castVec<T_Extents>(in);
            m_layout.outStrides = alpaka::fft::internal::castVec<T_Extents>(out);
            return *this;
        }

        /** Set the distance in elements between consecutive batches.
         *
         * Distances smaller than the addressed transform footprint can make batches overlap, which is only safe if
         * that aliasing is intentional. Call order relative to `extents()` does not matter.
         */
        PlanBuilder& distances(index_type inDistance, index_type outDistance)
        {
            m_layout.inDistance = inDistance;
            m_layout.outDistance = outDistance;
            return *this;
        }

        /** Request in-place execution.
         *
         * For real/complex transforms the buffer must use the padded FFT storage layout returned by the
         * `alpaka::fft::onHost::alloc*()` helpers or an equivalent manual layout.
         */
        PlanBuilder& inPlace()
        {
            m_options.placement = Placement::inPlace;
            return *this;
        }

        /** Request distinct input and output buffers. */
        PlanBuilder& outOfPlace()
        {
            m_options.placement = Placement::outOfPlace;
            return *this;
        }

        /** Let the backend allocate and own any temporary workspace. */
        PlanBuilder& backendManagedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::backendManaged;
            return *this;
        }

        /** Require the caller to provide workspace explicitly.
         *
         * Build succeeds only on backends that support external work areas. Use `workspaceBytes()` on the built
         * plan and call `setWorkspace()` before execution.
         */
        PlanBuilder& userProvidedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::userProvided;
            return *this;
        }

        /** Create a backend-specific plan for the selected device.
         *
         * The device selects the backend used for plan creation and later execution.
         */
        [[nodiscard]] auto build(alpaka::onHost::internal::concepts::Device auto& device) const
        {
            auto queue = device.makeQueue();
            return makePlanFromQueue(queue);
        }

    private:
        template<typename T_BuildQueue>
        requires alpaka::onHost::internal::concepts::Queue<T_BuildQueue>
        [[nodiscard]] auto makePlanFromQueue(T_BuildQueue& queue) const
        {
            using Api = decltype(alpaka::getApi(queue));
            return Plan<Api, T_Value, T_Extents>{queue, m_transform, m_layout, m_options};
        }

        Transform m_transform = Transform::c2c;
        Layout<T_Extents> m_layout{};
        PlanOptions m_options{};
    };

    /** Factory function to create a PlanBuilder with the given extents.
     *
     * Equivalent to `PlanBuilder<T_Value, T_Extents>{extents}` but shorter and more consistent with alpaka3
     * naming conventions.
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents = alpaka::fft::Extents<uint32_t, 1u>>
    [[nodiscard]] auto makePlan(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return PlanBuilder<T_Value, T_Extents>{extents};
    }

    /** Convenience wrapper for `Direction::forward`.
     *
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
    template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
    void executeForward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out)
    {
        plan.execute(queue, in, out, Direction::forward);
    }

    /** Convenience wrapper for `Direction::backward`.
     *
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
    template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
    void executeBackward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out)
    {
        plan.execute(queue, in, out, Direction::backward);
    }

    /** Run an in-place real-to-complex transform and return the complex view of the same storage.
     *
     * `buffer` is reinterpreted after execution; the returned view aliases the original allocation. Do not keep
     * using the old real extents as if they still described the transform output.
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
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

    /** Run an in-place complex-to-real transform and return the logical real view of the same storage.
     *
     * The returned view aliases `buffer` and exposes only the logical real extents, not the padded physical
     * storage that may still be present in memory.
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
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
