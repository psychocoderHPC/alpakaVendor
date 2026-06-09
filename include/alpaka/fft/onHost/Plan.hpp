/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/fft.hpp"
#include "alpaka/fft/onHost/alloc.hpp"

#include "SharedBufferFFT.hpp"

namespace alpaka::fft::onHost
{
    template<typename T_Value, std::size_t T_dim>
    class PlanBuilder;

    template<typename T_Api, typename T_Value, std::size_t T_dim>
    class Plan
    {
    public:
        using api_type = T_Api;
        using value_type = T_Value;
        static constexpr std::size_t dim = T_dim;

        Plan() = delete;

        Plan(auto& queue, Transform transform, Layout<T_dim> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
            , m_impl{queue, transform, layout, options}
        {
        }

        Plan(Plan const&) = delete;
        Plan& operator=(Plan const&) = delete;
        Plan(Plan&&) = default;
        Plan& operator=(Plan&&) = default;

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

        /**
         * Return the backend workspace size for this plan.
         *
         * For `WorkspacePolicy::userProvided` the caller must provide at least this many bytes with
         * `setWorkspace()` before the first execution. Some backends report `0` for transforms that do not
         * require temporary storage.
         */
        [[nodiscard]] auto workspaceBytes() const noexcept
        {
            return m_impl.workspaceBytes();
        }

        /**
         * Bind user-managed workspace to this plan.
         *
         * The storage must remain valid for every later execution that uses this plan. Rebinding replaces the
         * previously configured workspace for subsequent launches.
         */
        void setWorkspace(void* ptr, std::size_t bytes)
        {
            m_impl.setWorkspace(ptr, bytes);
        }

        /**
         * Execute the transform without implicit normalization.
         *
         * Forward and backward transforms use the raw vendor-library convention. A round-trip therefore usually
         * needs an explicit division by the logical FFT size to recover the original values.
         */
        template<typename T_In, typename T_Out>
        void execute(auto& queue, T_In const& in, T_Out& out, Direction direction)
        {
            m_impl.execute(queue, in, out, direction);
        }

    private:
        Transform m_transform;
        Layout<T_dim> m_layout;
        PlanOptions m_options;
        alpaka::fft::internal::PlanImpl<T_Api, T_Value, T_dim> m_impl;
    };

    template<typename T_Value, std::size_t T_dim>
    class PlanBuilder
    {
    public:
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

        /**
         * Set the logical transform extents.
         *
         * This also resets custom strides and batch distances back to contiguous defaults derived from the new
         * extents. Call `strides()` and `distances()` again afterwards if the plan should keep a custom layout.
         */
        PlanBuilder& extents(Extents<T_dim> value)
        {
            m_layout.extents = value;
            m_layout.inStrides = {};
            m_layout.outStrides = {};
            m_layout.inDistance = 0u;
            m_layout.outDistance = 0u;
            return *this;
        }

        /**
         * Set the number of transforms stored in the input/output buffers.
         *
         * Values below `1` are invalid and rejected during `build()`.
         */
        PlanBuilder& batch(std::size_t value)
        {
            m_layout.batch = value;
            return *this;
        }

        /**
         * Override the element strides passed to the backend.
         *
         * Strides are expressed in elements, not bytes, and follow alpaka's layout convention where the last
         * index is the fast-moving one.
         */
        PlanBuilder& strides(Strides<T_dim> in, Strides<T_dim> out)
        {
            m_layout.inStrides = in;
            m_layout.outStrides = out;
            return *this;
        }

        /**
         * Set the distance in elements between consecutive batches.
         *
         * Distances smaller than the addressed transform footprint can make batches overlap, which is only safe if
         * that aliasing is intentional.
         */
        PlanBuilder& distances(std::size_t inDistance, std::size_t outDistance)
        {
            m_layout.inDistance = inDistance;
            m_layout.outDistance = outDistance;
            return *this;
        }

        /**
         * Request in-place execution.
         *
         * For real/complex transforms the buffer must use the padded FFT storage layout returned by the
         * `alloc*ForFFT()` helpers or an equivalent manual layout.
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

        /**
         * Require the caller to provide workspace explicitly.
         *
         * Build succeeds only on backends that support external work areas. Use `workspaceBytes()` on the built
         * plan and call `setWorkspace()` before execution.
         */
        PlanBuilder& userProvidedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::userProvided;
            return *this;
        }

        /**
         * Create a backend-specific plan for the queue API.
         *
         * The queue selects the vendor backend and may also be used during plan creation, so any backend resources
         * associated with the queue must stay valid until the plan is no longer used.
         */
        [[nodiscard]] auto build(auto& queue) const
        {
            using Api = decltype(alpaka::getApi(queue));
            return Plan<Api, T_Value, T_dim>{queue, m_transform, m_layout, m_options};
        }

    private:
        Transform m_transform = Transform::c2c;
        Layout<T_dim> m_layout{};
        PlanOptions m_options{};
    };

    /** Convenience wrapper for `Direction::forward`. */
    template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
    void executeForward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out)
    {
        plan.execute(queue, in, out, Direction::forward);
    }

    /** Convenience wrapper for `Direction::backward`. */
    template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
    void executeBackward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out)
    {
        plan.execute(queue, in, out, Direction::backward);
    }

    /**
     * Run an in-place real-to-complex transform and return the complex view of the same storage.
     *
     * `buffer` is reinterpreted after execution; the returned view aliases the original allocation. Do not keep
     * using the old real extents as if they still described the transform output.
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

    /**
     * Run an in-place complex-to-real transform and return the logical real view of the same storage.
     *
     * The returned view aliases `buffer` and exposes only the logical real extents, not the padded physical
     * storage that may still be present in memory.
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
