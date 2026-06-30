/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <concepts>
#include <type_traits>

namespace alpakaVendor::test
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    template<typename T_Api>
    consteval bool isFftBackendEnabledForApi()
    {
        using Api = std::remove_cvref_t<T_Api>;
        if constexpr(std::same_as<Api, alpaka::api::Host>)
            return ALPAKAV_DEP_FFTW;
        else if constexpr(std::same_as<Api, alpaka::api::Cuda>)
            return ALPAKAV_DEP_CUFFT;
        else if constexpr(std::same_as<Api, alpaka::api::Hip>)
            return ALPAKAV_DEP_ROCFFT;
        else
            return false;
    }

    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    consteval bool isFftBackendEnabledForDevice(alpaka::onHost::Device<T_Api, T_DeviceKind> const&)
    {
        return isFftBackendEnabledForApi<T_Api>();
    }

    template<typename T_Type>
    inline void checkValue(T_Type actual, T_Type expected, double epsilon = 1.0e-5, double margin = 1.0e-6)
    {
        if constexpr(std::floating_point<T_Type>)
            CHECK(actual == Catch::Approx(expected).epsilon(epsilon).margin(margin));
        else
            CHECK(actual == expected);
    }
} // namespace alpakaVendor::test
