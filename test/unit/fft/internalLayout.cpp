/*
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: ISC
 */

#include "alpaka/fft/internal/utility.hpp"

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("expectedInputExtents c2c passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::c2c, Placement::outOfPlace) == Extents1D{8u});
    CHECK(internal::expectedInputExtents(layout, Transform::c2c, Placement::inPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE("expectedInputExtents r2c out-of-place passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::r2c, Placement::outOfPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE(
    "expectedInputExtents r2c in-place uses padded real extents",
    "[unit][helpers][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::r2c, Placement::inPlace) == Extents1D{10u});
}

TEMPLATE_LIST_TEST_CASE("expectedInputExtents c2r uses complex extents", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::c2r, Placement::outOfPlace) == Extents1D{5u});
    CHECK(internal::expectedInputExtents(layout, Transform::c2r, Placement::inPlace) == Extents1D{5u});
}

TEMPLATE_LIST_TEST_CASE("expectedOutputExtents c2c passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::c2c, Placement::outOfPlace) == Extents1D{8u});
    CHECK(internal::expectedOutputExtents(layout, Transform::c2c, Placement::inPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE("expectedOutputExtents r2c uses complex extents", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::r2c, Placement::outOfPlace) == Extents1D{5u});
    CHECK(internal::expectedOutputExtents(layout, Transform::r2c, Placement::inPlace) == Extents1D{5u});
}

TEMPLATE_LIST_TEST_CASE("expectedOutputExtents c2r out-of-place passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::c2r, Placement::outOfPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE(
    "expectedOutputExtents c2r in-place uses padded real extents",
    "[unit][helpers][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::c2r, Placement::inPlace) == Extents1D{10u});
}

TEMPLATE_LIST_TEST_CASE("expected strides are contiguous", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;
    using Strides1D = Strides<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInStrides(layout, Transform::c2c, Placement::outOfPlace) == Strides1D{1u});
    CHECK(internal::expectedOutStrides(layout, Transform::c2c, Placement::outOfPlace) == Strides1D{1u});
    CHECK(internal::expectedInStrides(layout, Transform::r2c, Placement::outOfPlace) == Strides1D{1u});
    CHECK(internal::expectedOutStrides(layout, Transform::r2c, Placement::outOfPlace) == Strides1D{1u});
    CHECK(internal::expectedInStrides(layout, Transform::c2r, Placement::outOfPlace) == Strides1D{1u});
    CHECK(internal::expectedOutStrides(layout, Transform::c2r, Placement::outOfPlace) == Strides1D{1u});
}

TEMPLATE_LIST_TEST_CASE("areZero detects zero and non-zero strides", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Strides1D = Strides<uint32_t, 1u>;
    using Strides2D = Strides<uint32_t, 2u>;

    CHECK(internal::areZero(Strides1D{0u}));
    CHECK_FALSE(internal::areZero(Strides1D{1u}));
    CHECK_FALSE(internal::areZero(Strides2D{0u, 1u}));
}

TEMPLATE_LIST_TEST_CASE("expected distances default to product of extents", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInDistance(layout, Transform::c2c, Placement::outOfPlace) == uint32_t{8u});
    CHECK(internal::expectedOutDistance(layout, Transform::c2c, Placement::outOfPlace) == uint32_t{8u});
    CHECK(internal::expectedInDistance(layout, Transform::r2c, Placement::outOfPlace) == uint32_t{8u});
    CHECK(internal::expectedOutDistance(layout, Transform::r2c, Placement::outOfPlace) == uint32_t{5u});
    CHECK(internal::expectedInDistance(layout, Transform::c2r, Placement::outOfPlace) == uint32_t{5u});
    CHECK(internal::expectedOutDistance(layout, Transform::c2r, Placement::outOfPlace) == uint32_t{8u});
}

TEMPLATE_LIST_TEST_CASE(
    "expected distances use custom layout distances when non-zero",
    "[unit][helpers][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}, .inDistance = uint32_t{16u}, .outDistance = uint32_t{16u}};

    CHECK(internal::expectedInDistance(layout, Transform::c2c, Placement::outOfPlace) == uint32_t{16u});
    CHECK(internal::expectedOutDistance(layout, Transform::c2c, Placement::outOfPlace) == uint32_t{16u});
}
