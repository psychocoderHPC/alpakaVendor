#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

project(
    alpakaVendor
    VERSION 0.1.0
    DESCRIPTION "Header-only vendor FFT abstraction on top of alpaka3."
    HOMEPAGE_URL "https://github.com/alpaka-group/alpakaVendor"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

option(alpakaV_TESTS "Build alpakaVendor tests." ${PROJECT_IS_TOP_LEVEL})
option(alpakaV_DEP_FFTW "Enable FFTW host backend." ON)
option(alpakaV_DEP_CUFFT "Enable cuFFT CUDA backend." OFF)
option(alpakaV_DEP_ROCFFT "Enable rocFFT HIP backend." OFF)
option(alpakaV_DOCS "Build documentation and code snippets." OFF)

if(alpaka_DEP_HIP)
    enable_language(HIP)
endif()
if(alpaka_DEP_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/alpaka3/CMakeLists.txt")
    add_subdirectory(extern/alpaka3)
else()
    FetchContent_Declare(
        alpaka3
        GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
        GIT_TAG 983b436c77b2be014aa654b9b2facf22bda8ad0e
    )
    FetchContent_MakeAvailable(alpaka3)
endif()

if(alpakaV_DEP_FFTW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3 fftw3f)
endif()

if(alpakaV_DEP_CUFFT)
    find_package(CUDAToolkit REQUIRED)
endif()

if(alpakaV_DEP_ROCFFT)
    find_package(
        rocfft
        REQUIRED
        CONFIG
        HINTS "$ENV{ROCM_PATH}" /opt/rocm /opt/rocm-6.3.4
        PATH_SUFFIXES lib/cmake/rocfft
    )
endif()

add_library(alpakaVendor INTERFACE)
add_library(alpakaVendor::alpakaVendor ALIAS alpakaVendor)

target_compile_features(alpakaVendor INTERFACE cxx_std_20)
target_include_directories(
    alpakaVendor
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(alpakaVendor INTERFACE $<BUILD_INTERFACE:alpaka::alpaka>)
target_compile_definitions(
    alpakaVendor
    INTERFACE
        ALPAKAV_DEP_FFTW=$<BOOL:${alpakaV_DEP_FFTW}>
        ALPAKAV_DEP_CUFFT=$<BOOL:${alpakaV_DEP_CUFFT}>
        ALPAKAV_DEP_ROCFFT=$<BOOL:${alpakaV_DEP_ROCFFT}>
)
if(alpakaV_DEP_FFTW)
    target_link_libraries(alpakaVendor INTERFACE PkgConfig::FFTW3)
endif()
if(alpakaV_DEP_CUFFT)
    target_link_libraries(alpakaVendor INTERFACE CUDA::cufft)
endif()
if(alpakaV_DEP_ROCFFT)
    target_link_libraries(alpakaVendor INTERFACE roc::rocfft)
endif()

if(alpakaV_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()

if(alpakaV_DOCS)
    add_subdirectory(doc)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS alpakaVendor EXPORT alpakaVendorTargets)
install(EXPORT alpakaVendorTargets NAMESPACE alpakaVendor:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaVendor)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/alpakaVendorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/alpakaVendorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/alpakaVendorConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaVendor
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/alpakaVendorConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/alpakaVendorConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaVendor
)
