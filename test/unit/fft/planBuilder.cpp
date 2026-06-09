/*
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("PlanBuilder defaults are applied", "[unit][plan]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    auto exec = getExecutor(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, 1>{}.c2c().extents({8u}).build(queue);
        CHECK(plan.transform() == Transform::c2c);
        CHECK(plan.layout().extents == Extents<1>{8u});
        CHECK(plan.options().placement == Placement::outOfPlace);
        CHECK(plan.options().workspacePolicy == WorkspacePolicy::backendManaged);
    }
}

TEMPLATE_LIST_TEST_CASE("invalid plan configuration throws", "[unit][plan]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    auto exec = getExecutor(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();

        CHECK_THROWS_AS(
            (alpaka::fft::onHost::PlanBuilder<float, 1>{}.r2c().extents({0u}).build(queue)),
            std::invalid_argument);
        CHECK_THROWS_AS(
            (alpaka::fft::onHost::PlanBuilder<Complex, 1>{}.r2c().extents({8u}).build(queue)),
            std::invalid_argument);
    }
}
