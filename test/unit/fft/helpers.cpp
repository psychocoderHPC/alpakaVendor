/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "alpaka/fft/internal/utility.hpp"

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("fft helper extents", "[unit][helpers]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint16_t, 2u>;
    using Strides2D = Strides<uint16_t, 2u>;

    CHECK(r2cComplexExtent(uint32_t{8u}) == uint32_t{5u});
    CHECK(r2cComplexExtent(uint32_t{9u}) == uint32_t{5u});
    CHECK(r2cPaddedRealExtent(uint32_t{8u}) == uint32_t{10u});
    CHECK(r2cPaddedRealExtent(uint32_t{10u}) == uint32_t{12u});

    constexpr auto logical = Extents2D{4u, 9u};
    CHECK(r2cComplexExtent(logical) == Extents2D{4u, 5u});
    CHECK(r2cPaddedRealExtent(logical) == Extents2D{4u, 10u});
    CHECK(product(logical) == uint16_t{36u});
    CHECK(contiguousStrides(logical) == Strides2D{9u, 1u});
}

TEMPLATE_LIST_TEST_CASE("in place real storage descriptor", "[unit][helpers]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto storage = makeInPlaceRealStorage<float>(Extents2D{4u, 8u});
    CHECK(storage.logicalRealExtents == Extents2D{4u, 8u});
    CHECK(storage.physicalRealExtents == Extents2D{4u, 10u});
    CHECK(storage.logicalComplexExtents == Extents2D{4u, 5u});
    CHECK(storage.logicalRealElements == uint32_t{32u});
    CHECK(storage.physicalRealElements == uint32_t{40u});
    CHECK(storage.logicalComplexElements == uint32_t{20u});
}

TEMPLATE_LIST_TEST_CASE(
    "internal FFT layout helpers use padded real extents for in-place C2R",
    "[unit][helpers]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::c2r, Placement::inPlace) == Extents1D{5u});
    CHECK(internal::expectedOutputExtents(layout, Transform::c2r, Placement::inPlace) == Extents1D{10u});
    CHECK(internal::expectedInDistance(layout, Transform::c2r, Placement::inPlace) == uint32_t{5u});
    CHECK(internal::expectedOutDistance(layout, Transform::c2r, Placement::inPlace) == uint32_t{10u});
}
