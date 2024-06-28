cmake_minimum_required(VERSION 3.16)

project(tt-conv-based-on-metalium)

find_program(CLANGPP_17 clang++-17)
find_program(CLANG_17 clang-17)

if (NOT CLANGPP_17)
    message(FATAL_ERROR "clang++-17 not found")
endif()

if (NOT CLANG_17)
    message(FATAL_ERROR "clang-17 not found")
endif()

set(CMAKE_CXX_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_ID "Clang")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
if(${CMAKE_CXX_STANDARD} LESS 20)
  message(FATAL_ERROR "tt-conv-based-on-metalium requires C++20 or later, but is configured for C++${CMAKE_CXX_STANDARD})")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

option(ENABLE_RTTI "Enables RTTI" OFF)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

set(TT_METAL_DIR ${PROJECT_SOURCE_DIR}/external/tt-metal)

add_subdirectory(external/tt-metal)
add_subdirectory(src)

# Temporary workaround for Issue #8767
install(DIRECTORY ${PROJECT_BINARY_DIR}/external/tt-metal/hw/toolchain
    DESTINATION ${TT_METAL_DIR}/runtime/hw
)
