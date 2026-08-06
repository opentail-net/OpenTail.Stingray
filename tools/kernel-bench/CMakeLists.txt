cmake_minimum_required(VERSION 3.20)
project(opentail_kernel_bench CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link llama.cpp's real ggml rather than copying kernel source. A copied kernel that lost a
# #define or picked up different struct packing gives a clean number for the wrong thing —
# and "measured something other than intended" is the failure this comparison exists to avoid.
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../LLamaSharp/llama.cpp"
    CACHE PATH "Path to a llama.cpp source checkout")

# Build only what is needed.
set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# CRITICAL: ggml normally compiles ggml-cpu several times for different ISA levels and dispatches
# at runtime. If the harness resolves the GENERIC variant it benchmarks the scalar reference and
# makes llama.cpp look slow — inverting the conclusion. Pin a single native build so the symbol
# we call is the vectorised one. main.cpp also cross-checks arch vs _generic at runtime and
# refuses to report if they are the same speed.
set(GGML_NATIVE       ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
# The harness intentionally calls low-level CPU quantization/dot functions that are not
# exported by ggml's shared-library ABI. Link the native ggml objects statically instead.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}/ggml" ggml_build EXCLUDE_FROM_ALL)

add_executable(kernel-bench main.cpp)
target_link_libraries(kernel-bench PRIVATE ggml ggml-base ggml-cpu)
target_include_directories(kernel-bench PRIVATE
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu")
