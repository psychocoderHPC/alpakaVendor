#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

alpaka_install_catch2()
function(alpakaV_add_test_target target_prefix folder)
    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS "${folder}/*.cpp")
    foreach(source_file ${test_sources})
        get_filename_component(source_file_name ${source_file} NAME_WE)
        get_filename_component(test_folder_name ${source_file} DIRECTORY)
        get_filename_component(test_folder_name ${test_folder_name} NAME)

        set(test_target_name "${target_prefix}${test_folder_name}_${source_file_name}")
        add_executable(${test_target_name} ${source_file})
        target_link_libraries(
            ${test_target_name}
            PRIVATE Catch2::Catch2WithMain alpakaVendor::alpakaVendor alpaka::alpaka
        )
        target_include_directories(
            ${test_target_name}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        if(NOT alpakaV_DEP_FFTW)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Host_Cpu ALPAKA_DISABLE_Host_NumaCpu)
        endif()
        if(NOT alpakaV_DEP_CUFFT)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
        endif()
        if(NOT alpakaV_DEP_ROCFFT)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
        endif()
        if(NOT alpakaV_DEP_ONEMKL)
            target_compile_definitions(
                ${test_target_name}
                PRIVATE
                    ALPAKA_DISABLE_OneApi_Cpu
                    ALPAKA_DISABLE_OneApi_IntelGpu
                    ALPAKA_DISABLE_OneApi_NvidiaGpu
                    ALPAKA_DISABLE_OneApi_AmdGpu
            )
        endif()
        alpaka_finalize(${test_target_name})
        catch_discover_tests(${test_target_name})
    endforeach()
endfunction()

alpakav_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit)
alpakav_add_test_target(integr_test_ ${CMAKE_CURRENT_SOURCE_DIR}/integr)
