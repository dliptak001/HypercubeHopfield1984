cmake_minimum_required(VERSION 3.20)
project(HypercubeHopfield1984Python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── Optimization flags (match the main project / sibling python builds) ──
# HYPERCUBE_ARCH controls -march. Defaults to "native" for local dev builds.
# cibuildwheel overrides to "x86-64-v2" (x86_64) or "none" (ARM, MSVC).
set(HYPERCUBE_ARCH "native" CACHE STRING "Target architecture for -march (native, x86-64-v2, none)")

if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -ffast-math)
    if(NOT HYPERCUBE_ARCH STREQUAL "none")
        if(HYPERCUBE_ARCH STREQUAL "native")
            add_compile_options(-march=native -mtune=native)
        else()
            add_compile_options(-march=${HYPERCUBE_ARCH} -mtune=generic)
        endif()
    endif()
    add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

# ── Core source compiled directly into the module (PIC required) ──
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CORE_SOURCES
    ${CORE_DIR}/Hopfield.cpp
)

# ── Package version (single source: hypercube_hopfield1984/_version.py) ──
set(_HH84_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/hypercube_hopfield1984/_version.py")
set(HYPERCUBE_HOPFIELD1984_VERSION "")
file(STRINGS "${_HH84_VERSION_FILE}" _HH84_VERSION_LINES)
foreach(_line IN LISTS _HH84_VERSION_LINES)
    # Strip CR (Windows) and match: __version__ = "x.y.z"
    string(REPLACE "\r" "" _line "${_line}")
    if(_line MATCHES "^__version__[ \t]*=[ \t]*\"([^\"]+)\"")
        set(HYPERCUBE_HOPFIELD1984_VERSION "${CMAKE_MATCH_1}")
        break()
    endif()
endforeach()
if(HYPERCUBE_HOPFIELD1984_VERSION STREQUAL "")
    message(FATAL_ERROR
        "Could not parse __version__ from hypercube_hopfield1984/_version.py")
endif()
message(STATUS "hypercube_hopfield1984 version: ${HYPERCUBE_HOPFIELD1984_VERSION}")

# ── Build the Python extension module ──
pybind11_add_module(_core bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE ${CORE_DIR})
target_compile_definitions(_core PRIVATE
    "HYPERCUBE_HOPFIELD1984_VERSION=\"${HYPERCUBE_HOPFIELD1984_VERSION}\"")

# ── Linking ──
if(MINGW)
    # Static libgcc/libstdc++ keep the .pyd free of those DLLs. Do NOT static-link
    # winpthread: mingw-w64 15.2.0's libwinpthread.a references __intrinsic_setjmpex
    # and fails at link (same constraint as the sibling python builds).
    # Posix-model MinGW still needs libwinpthread-1.dll at runtime — ship the DLL
    # from THIS toolchain (stale copies miss symbols like nanosleep64).
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    find_file(WINPTHREAD_DLL libwinpthread-1.dll PATHS ENV PATH NO_DEFAULT_PATH)
    if(WINPTHREAD_DLL)
        install(FILES ${WINPTHREAD_DLL} DESTINATION hypercube_hopfield1984)
    endif()
endif()

# ── Install into the hypercube_hopfield1984 package directory ──
install(TARGETS _core DESTINATION hypercube_hopfield1984)
