cmake_minimum_required(VERSION 3.18)
project(NeuronDummyPluginEP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths – set via -D on the command line or edit here.
# ---------------------------------------------------------------------------

# Where the onnxruntime headers live.
# Default: look next to this CMakeLists.txt for an "onnxruntime" subdirectory
# (useful if you copy the install tree next to the source).
# Override: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime_install ..
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime_install")
endif()

# Where to find libonnxruntime.so / onnxruntime.lib.
# The install layout is <root>/lib/ on Linux and <root>/lib/ on Windows.
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
# MODULE on Linux/macOS  → .so with no import library (correct for plugin EPs)
# SHARED on Windows      → .dll
if(WIN32)
  add_library(neuron_dummy_plugin_ep SHARED ${NEURON_EP_SRCS})
else()
  add_library(neuron_dummy_plugin_ep MODULE ${NEURON_EP_SRCS})
endif()

# Public ORT headers
target_include_directories(neuron_dummy_plugin_ep PRIVATE
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    "${GSL_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}"   # so #include "neuron_ep_utils.h" works
)

# Link against onnxruntime shared library.
# find_library() honours CMAKE_FIND_ROOT_PATH when cross-compiling.
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_LIB_DIR}"
    REQUIRED)

target_link_libraries(neuron_dummy_plugin_ep PRIVATE ${ONNXRUNTIME_LIB})

# ---------------------------------------------------------------------------
# Symbol visibility (hide everything except CreateEpFactories/ReleaseEpFactory)
# ---------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
    # Linux – use version script
    target_link_options(neuron_dummy_plugin_ep PRIVATE
        "-Xlinker" "--version-script=${CMAKE_SOURCE_DIR}/neuron_dummy_plugin_ep.lds"
        "-Xlinker" "--no-undefined"
        "-Xlinker" "--gc-sections"
        "-z" "noexecstack"
    )
elseif(APPLE)
    # macOS – EXPORT_SYMBOL macro in the source handles visibility.
    # Also strip dead code.
    target_link_options(neuron_dummy_plugin_ep PRIVATE
        "-Xlinker" "-dead_strip"
    )
elseif(WIN32)
    # Windows – .def file controls exports.
    target_link_options(neuron_dummy_plugin_ep PRIVATE
        "-DEF:${CMAKE_SOURCE_DIR}/neuron_dummy_plugin_ep.def"
    )
endif()

# Default symbol visibility: hidden (no leaking of C++ internals).
set_target_properties(neuron_dummy_plugin_ep PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(neuron_dummy_plugin_ep PRIVATE
        -Wall -Wextra -Wno-unused-parameter
    )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS neuron_dummy_plugin_ep
        LIBRARY  DESTINATION lib
        RUNTIME  DESTINATION bin)

message(STATUS "NeuronDummyPluginEP build:")
message(STATUS "  ORT root       : ${ONNXRUNTIME_ROOT}")
message(STATUS "  ORT lib dir    : ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "  GSL include    : ${GSL_INCLUDE_DIR}")
message(STATUS "  ORT library    : ${ONNXRUNTIME_LIB}")
message(STATUS "  Target triple  : ${CMAKE_SYSTEM_PROCESSOR}")
