cmake_minimum_required(VERSION 3.18)
project(NeuronPluginEP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths – set via -D on the command line or edit here.
# ---------------------------------------------------------------------------

# Where the onnxruntime headers live.
# Default: look next to this CMakeLists.txt for an "onnxruntime_install" subdirectory.
# Override: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime_install ..
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime_install")
endif()

# Where to find libonnxruntime.so.
if(NOT DEFINED ONNXRUNTIME_LIB_DIR)
  set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
endif()

# GSL (Microsoft.GSL) include directory.
# Override: cmake -DGSL_INCLUDE_DIR=/path/to/gsl/include ..
if(NOT DEFINED GSL_INCLUDE_DIR)
  set(GSL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/gsl/include")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(NEURON_EP_SRCS
    neuron_dummy_plugin_ep.cc
    neuron_ep_factory.cc
    neuron_ep.cc
    neuron_ep_data_transfer.cc
    neuron_ep_stream_support.cc
)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
# MODULE → .so with no import library (correct for plugin EPs).
add_library(onnxruntime_providers_neuron MODULE ${NEURON_EP_SRCS})

# Public ORT headers
target_include_directories(onnxruntime_providers_neuron PRIVATE
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    "${GSL_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}"   # so #include "neuron_ep_utils.h" works
)

# Link dl for dlopen (used to load libonnxruntime_provider_neuron_wrapper.so).
target_link_libraries(onnxruntime_providers_neuron PRIVATE dl)

# Link against onnxruntime shared library (optional).
# A plugin EP is loaded by ORT at runtime via dlopen(), so ORT symbols are
# already present in the process. We only link if the .so is found, otherwise
# we rely on runtime resolution (--allow-shlib-undefined).
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_LIB_DIR}"
)

if(ONNXRUNTIME_LIB)
    target_link_libraries(onnxruntime_providers_neuron PRIVATE ${ONNXRUNTIME_LIB})
    message(STATUS "  Linking against : ${ONNXRUNTIME_LIB}")
else()
    message(STATUS "  libonnxruntime.so not found - relying on runtime symbol resolution")
    target_link_options(onnxruntime_providers_neuron PRIVATE
        "-Wl,--allow-shlib-undefined"
    )
endif()

# ---------------------------------------------------------------------------
# Symbol visibility (hide everything except CreateEpFactories/ReleaseEpFactory)
# ---------------------------------------------------------------------------
target_link_options(onnxruntime_providers_neuron PRIVATE
    "-Wl,--version-script=${CMAKE_SOURCE_DIR}/neuron_dummy_plugin_ep.lds"
    "-Wl,--no-undefined"
    "-Wl,--gc-sections"
    "-Wl,-z,noexecstack"
)

set_target_properties(onnxruntime_providers_neuron PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(onnxruntime_providers_neuron PRIVATE
        -Wall -Wextra -Wno-unused-parameter
    )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS onnxruntime_providers_neuron
        LIBRARY DESTINATION lib)

message(STATUS "NeuronPluginEP build:")
message(STATUS "  ORT root       : ${ONNXRUNTIME_ROOT}")
message(STATUS "  ORT lib dir    : ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "  GSL include    : ${GSL_INCLUDE_DIR}")
message(STATUS "  ORT library    : ${ONNXRUNTIME_LIB}")
message(STATUS "  Target triple  : ${CMAKE_SYSTEM_PROCESSOR}")
