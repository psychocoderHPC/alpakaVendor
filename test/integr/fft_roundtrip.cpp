/*
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>
#include <numbers>

#include "../unit/test.hpp"
#include "alpaka/fft.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("FFT C2C roundtrip 1D", "[integr][fft][c2c]", TestBackends)
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

        auto queue = device.makeQueue();
        constexpr std::size_t n = 8u;
        auto extents = alpaka::Vec<std::size_t, 1u>{n};

        // Use unified memory for host-side data access
        auto in = alpaka::fft::onHost::allocUnifiedForFFT<Complex>(device, extents);
        auto tmp = alpaka::fft::onHost::allocUnifiedForFFT<Complex>(device, extents);
        auto out = alpaka::fft::onHost::allocUnifiedForFFT<Complex>(device, extents);

        for(std::size_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(static_cast<int>(i) - 3), float(static_cast<int>(i % 3u) - 1)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, 1>{}.c2c().extents({n}).build(queue);
        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(std::size_t i = 0; i < n; ++i)
        {
            CHECK(
                out.data()[i].real() == Catch::Approx(in.data()[i].real() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
            CHECK(
                out.data()[i].imag() == Catch::Approx(in.data()[i].imag() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("FFT C2C roundtrip 1D accepts plain alpaka buffers", "[integr][fft][c2c][plain]", TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;

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

        auto queue = device.makeQueue();
        constexpr std::size_t n = 8u;
        auto extents = alpaka::Vec<std::size_t, 1u>{n};

        auto in = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::onHost::allocUnified<Complex>(device, extents);

        for(std::size_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(static_cast<int>(i) - 3), float(static_cast<int>(i % 3u) - 1)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, 1>{}.c2c().extents({n}).build(queue);
        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(std::size_t i = 0; i < n; ++i)
        {
            CHECK(
                out.data()[i].real() == Catch::Approx(in.data()[i].real() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
            CHECK(
                out.data()[i].imag() == Catch::Approx(in.data()[i].imag() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT R2C/C2R out-of-place accepts plain alpaka buffers",
    "[integr][fft][r2c][c2r][plain]",
    TestBackends)
{
    using namespace alpaka::fft;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;

        auto queue = device.makeQueue();
        constexpr std::size_t n = 8u;
        constexpr std::size_t complexN = r2cComplexExtent(n);

        auto in = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<std::size_t, 1u>{n});
        auto spectrum = alpaka::onHost::allocUnified<alpaka::math::Complex<float>>(
            device,
            alpaka::Vec<std::size_t, 1u>{complexN});
        auto out = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<std::size_t, 1u>{n});

        for(std::size_t i = 0; i < n; ++i)
            in.data()[i] = float(i + 1u);

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float, 1>{}.r2c().extents({n}).outOfPlace().build(queue);
        alpaka::fft::onHost::executeForward(queue, r2cPlan, in, spectrum);

        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float, 1>{}.c2r().extents({n}).outOfPlace().build(queue);
        alpaka::fft::onHost::executeBackward(queue, c2rPlan, spectrum, out);
        alpaka::onHost::wait(queue);

        for(std::size_t i = 0; i < n; ++i)
            CHECK(out.data()[i] == Catch::Approx(float(i + 1u) * float(n)).epsilon(1.0e-4));
    }
}

TEMPLATE_LIST_TEST_CASE("FFT R2C/C2R in place 1D", "[integr][fft][r2c][c2r]", TestBackends)
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

        auto queue = device.makeQueue();
        constexpr std::size_t n = 8u;
        auto storage = makeInPlaceRealStorage<float>(Extents<1>{n});
        // Use unified memory for host-side data access
        auto realBuffer = alpaka::fft::onHost::allocUnifiedForFFT<float>(device, alpaka::Vec<std::size_t, 1u>{n});

        for(std::size_t i = 0; i < n; ++i)
            realBuffer.data()[i] = float(i + 1u);
        for(std::size_t i = n; i < storage.physicalRealElements; ++i)
            realBuffer.data()[i] = 0.0f;

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float, 1>{}.r2c().extents({n}).inPlace().build(queue);
        auto complexBuffer = alpaka::fft::onHost::executeR2CInPlace(queue, r2cPlan, realBuffer);
        CHECK(alpaka::fft::internal::toExtents<1>(complexBuffer.getExtents()) == storage.logicalComplexExtents);

        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float, 1>{}.c2r().extents({n}).inPlace().build(queue);
        auto recovered = alpaka::fft::onHost::executeC2RInPlace(queue, c2rPlan, complexBuffer);
        alpaka::onHost::wait(queue);
        CHECK(alpaka::fft::internal::toExtents<1>(recovered.getExtents()) == Extents<1>{n});
        for(std::size_t i = 0; i < n; ++i)
            CHECK(recovered.data()[i] == Catch::Approx(float(i + 1u) * float(n)).epsilon(1.0e-4));
    }
}
