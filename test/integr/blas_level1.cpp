/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>

#include "../unit/blas/reference.hpp"
#include "../unit/test.hpp"
#include "alpaka/blas.hpp"

using namespace alpakaVendor::test;

template<typename T>
void fillVector(T* ptr, std::size_t n)
{
    for(std::size_t i = 0; i < n; ++i)
    {
        if constexpr(alpaka::blas::ComplexScalar<T>)
            ptr[i] = T{static_cast<typename T::value_type>(i + 1), static_cast<typename T::value_type>(2 * i + 1)};
        else
            ptr[i] = T(i + 1);
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS level1 real and complex vectors", "[integr][blas][level1]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t n = 5u;
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        auto x = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto y = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto z = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto dotResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto nrm2Result = alpaka::onHost::allocUnified<float>(device, 1u);
        auto asumResult = alpaka::onHost::allocUnified<float>(device, 1u);
        auto iamaxResult = alpaka::onHost::allocUnified<int>(device, 1u);

        fillVector(x.data(), n);
        fillVector(y.data(), n);
        for(uint32_t i = 0; i < n; ++i)
            z.data()[i] = {};
        auto xBefore = alpaka::onHost::allocHostLike(x);
        auto yBefore = alpaka::onHost::allocHostLike(y);
        std::copy_n(x.data(), n, xBefore.data());
        std::copy_n(y.data(), n, yBefore.data());

        alpaka::blas::onHost::copy(queue, x, z, options);
        alpaka::blas::onHost::swap(queue, x, y, options);
        alpaka::blas::onHost::scal(queue, Scalar{2, -1}, x, options);
        alpaka::blas::onHost::axpy(queue, Scalar{2, -1}, x, y, options);
        alpaka::blas::onHost::dot(queue, x, z, dotResult, options);
        alpaka::blas::onHost::nrm2(queue, x, nrm2Result, options);
        alpaka::blas::onHost::asum(queue, x, asumResult, options);
        alpaka::blas::onHost::iamax(queue, x, iamaxResult, options);
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(z.data()[i] == xBefore.data()[i]);
            CHECK(y.data()[i] == xBefore.data()[i] + Scalar{2, -1} * x.data()[i]);
            CHECK(x.data()[i] == Scalar{2, -1} * yBefore.data()[i]);
        }
        auto const dotExpected = blas::dotRef(x.data(), z.data(), n);
        CHECK(dotResult.data()[0].real() == Catch::Approx(dotExpected.real()).epsilon(1e-4));
        CHECK(dotResult.data()[0].imag() == Catch::Approx(dotExpected.imag()).epsilon(1e-4));
        CHECK(nrm2Result.data()[0] == Catch::Approx(blas::nrm2Ref(x.data(), n)).epsilon(1e-4));
        CHECK(asumResult.data()[0] == Catch::Approx(blas::asumRef(x.data(), n)).epsilon(1e-4));
        CHECK(iamaxResult.data()[0] == blas::iamaxRef(x.data(), n));
    }
}
