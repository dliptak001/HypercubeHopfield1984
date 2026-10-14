cmake_minimum_required(VERSION 3.20)
project(HypercubeHopfield1984 VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Family convention (matches HypercubeLCN): fast-math in Release, everywhere,
# so the exes and the Python wheel compile the same way. HYPERCUBE_ARCH
# controls -march (same knob as python/CMakeLists.txt). "native" for local
# dev builds; release packaging pins "x86-64-v2"; "none" skips -march (ARM, MSVC).
set(HYPERCUBE_ARCH "native" CACHE STRING
    "Target architecture for -march (native, x86-64-v2, none)")

if(MSVC)
    add_compile_options("$<$<CONFIG:Release>:/O2;/fp:fast>")
else()
    add_compile_options("$<$<CONFIG:Release>:-O3;-ffast-math>")
    if(NOT HYPERCUBE_ARCH STREQUAL "none")
        if(HYPERCUBE_ARCH STREQUAL "native")
            add_compile_options(
                "$<$<CONFIG:Release>:-march=native;-mtune=native>")
        else()
            add_compile_options(
                "$<$<CONFIG:Release>:-march=${HYPERCUBE_ARCH};-mtune=generic>")
        endif()
    endif()
    add_compile_options(-Wall -Wextra)
endif()

# The SDK surface: the one class. Examples, tests, and the Python bindings
# all link this.
add_library(HypercubeHopfield1984Core STATIC Hopfield.cpp)
target_include_directories(HypercubeHopfield1984Core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})

# The example and the benchmark default to the bundled exemplars; the path
# is compiled in as an absolute path into this source tree.
set(HOPFIELD_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(hopfield_example examples/example.cpp)
target_link_libraries(hopfield_example PRIVATE HypercubeHopfield1984Core)
target_compile_definitions(hopfield_example PRIVATE
        "HOPFIELD_DATA_DIR=\"${HOPFIELD_DATA_DIR}\"")

enable_testing()

add_executable(hopfield_smoketest test/smoketest.cpp)
target_link_libraries(hopfield_smoketest PRIVATE HypercubeHopfield1984Core)
add_test(NAME smoketest COMMAND hopfield_smoketest)

add_executable(hopfield_recall_benchmark examples/recall_benchmark.cpp)
target_link_libraries(hopfield_recall_benchmark PRIVATE HypercubeHopfield1984Core)
target_compile_definitions(hopfield_recall_benchmark PRIVATE
        "HOPFIELD_DATA_DIR=\"${HOPFIELD_DATA_DIR}\"")
