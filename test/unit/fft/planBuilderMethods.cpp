/*
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("PlanBuilder batch sets layout batch", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().batch(4u).build(device);
        CHECK(plan.layout().batch == uint32_t{4u});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder distances sets layout distances", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().distances(16u, 16u).build(device);
        CHECK(plan.layout().inDistance == uint32_t{16u});
        CHECK(plan.layout().outDistance == uint32_t{16u});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder strides sets layout strides", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;
        auto inStrides = Strides<uint32_t, 1u>{1u};
        auto outStrides = Strides<uint32_t, 1u>{1u};
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().strides(inStrides, outStrides).build(device);
        CHECK(plan.layout().inStrides == Strides<uint32_t, 1u>{1u});
        CHECK(plan.layout().outStrides == Strides<uint32_t, 1u>{1u});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder invalid batch zero throws", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        CHECK_THROWS_AS(
            (alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().batch(0u).build(device)),
            std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder complex type with r2c throws", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        CHECK_THROWS_AS((alpaka::fft::onHost::PlanBuilder<Complex>{8u}.r2c().build(device)), std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder complex type with c2r throws", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        CHECK_THROWS_AS((alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2r().build(device)), std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("makePlan factory produces working builder", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::makePlan<Complex>(8u).c2c().build(device);
        CHECK(plan.transform() == Transform::c2c);
        CHECK(plan.layout().extents == Extents<uint32_t, 1u>{8u});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder r2c with real type is valid", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{8u}.r2c().build(device);
        CHECK(plan.transform() == Transform::r2c);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder c2r with real type is valid", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{8u}.c2r().build(device);
        CHECK(plan.transform() == Transform::c2r);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "PlanBuilder options default to outOfPlace and backendManaged",
    "[unit][plan][methods]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().build(device);
        CHECK(plan.options().placement == Placement::outOfPlace);
        CHECK(plan.options().workspacePolicy == WorkspacePolicy::backendManaged);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder inPlace sets placement", "[unit][plan][methods]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{8u}.r2c().inPlace().build(device);
        CHECK(plan.options().placement == Placement::inPlace);
    }
}
