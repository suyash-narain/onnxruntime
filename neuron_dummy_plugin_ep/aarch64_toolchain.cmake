# aarch64_toolchain.cmake
# CMake toolchain file for cross-compiling to aarch64-linux-gnu.
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=aarch64_toolchain.cmake \
#         -DONNXRUNTIME_ROOT=/path/to/ort-aarch64-install \
#         -DGSL_INCLUDE_DIR=/path/to/gsl/include \
#         ..

# ---------------------------------------------------------------------------
# Target system
# ---------------------------------------------------------------------------
set(CMAKE_SYSTEM_NAME    Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ---------------------------------------------------------------------------
# Cross-compiler
# Install on Ubuntu/Debian: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
# ---------------------------------------------------------------------------
set(CROSS_COMPILER_PREFIX "aarch64-linux-gnu")

find_program(CMAKE_C_COMPILER   "${CROSS_COMPILER_PREFIX}-gcc"   REQUIRED)
find_program(CMAKE_CXX_COMPILER "${CROSS_COMPILER_PREFIX}-g++"   REQUIRED)
find_program(CMAKE_STRIP        "${CROSS_COMPILER_PREFIX}-strip"  REQUIRED)
find_program(CMAKE_AR           "${CROSS_COMPILER_PREFIX}-ar"     REQUIRED)
find_program(CMAKE_RANLIB       "${CROSS_COMPILER_PREFIX}-ranlib" REQUIRED)

# ---------------------------------------------------------------------------
# Sysroot (optional but recommended for finding target libraries).
# Leave empty if you don't have a full sysroot; the cross-compiler's
# default multilib layout is usually sufficient for our use case.
# ---------------------------------------------------------------------------
# set(CMAKE_SYSROOT "/path/to/aarch64-sysroot")

# ---------------------------------------------------------------------------
# Search rules: only look in target paths for libraries and headers.
# ---------------------------------------------------------------------------
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # host tools (cmake, etc.)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # target libraries
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)    # target headers
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---------------------------------------------------------------------------
# C++ standard settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Useful for debugging cross-compile issues.
# set(CMAKE_VERBOSE_MAKEFILE ON)
