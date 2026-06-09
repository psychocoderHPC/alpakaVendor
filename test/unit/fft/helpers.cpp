/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("fft helper extents", "[unit][helpers]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    CHECK(r2cComplexExtent(8u) == 5u);
    CHECK(r2cComplexExtent(9u) == 5u);
    CHECK(r2cPaddedRealExtent(8u) == 10u);
    CHECK(r2cPaddedRealExtent(10u) == 12u);

    constexpr auto logical = Extents<2>{4u, 9u};
    CHECK(r2cLogicalComplexExtents(logical) == Extents<2>{4u, 5u});
    CHECK(r2cInPlaceRealStorageExtents(logical) == Extents<2>{4u, 10u});
    CHECK(product(logical) == 36u);
    CHECK(contiguousStrides(logical) == Strides<2>{9u, 1u});
}

TEMPLATE_LIST_TEST_CASE("in place real storage descriptor", "[unit][helpers]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    auto storage = makeInPlaceRealStorage<float>(Extents<2>{4u, 8u});
    CHECK(storage.logicalRealExtents == Extents<2>{4u, 8u});
    CHECK(storage.physicalRealExtents == Extents<2>{4u, 10u});
    CHECK(storage.logicalComplexExtents == Extents<2>{4u, 5u});
    CHECK(storage.logicalRealElements == 32u);
    CHECK(storage.physicalRealElements == 40u);
    CHECK(storage.logicalComplexElements == 20u);
}
