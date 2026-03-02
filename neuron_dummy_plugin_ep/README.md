# NeuronDummyPluginEP

A minimal **OnnxRuntime plugin execution provider** for Linux (x86_64 and aarch64).
It claims all `Add` nodes with static, equal-shaped float32 inputs and runs them on the CPU.
The purpose is to exercise the full ORT plugin EP API surface (factory, allocator, data transfer, stream/notification) without any real hardware dependency.

---

## File overview

### Build system

| File | What it does |
|---|---|
| `CMakeLists.txt` | Builds `neuron_dummy_plugin_ep.so` as a `MODULE` library. Finds the ORT and GSL headers, links against `libonnxruntime.so` when present (falls back to runtime symbol resolution otherwise), applies the version script, and installs to `lib/`. |
| `aarch64_toolchain.cmake` | CMake toolchain file for cross-compiling to `aarch64-linux-gnu`. Pass via `-DCMAKE_TOOLCHAIN_FILE=aarch64_toolchain.cmake`. |

### Symbol export control

| File | What it does |
|---|---|
| `neuron_dummy_plugin_ep.lds` | Linux linker version script. Exports only `CreateEpFactories` and `ReleaseEpFactory`; hides everything else to prevent symbol collisions when multiple plugin EPs are loaded in the same process. |
| `neuron_dummy_plugin_ep.def` | Legacy Windows DEF file listing the same two exports. Not used by the Linux build. |

### C++ source

| File | What it does |
|---|---|
| `neuron_dummy_plugin_ep.cc` | Shared-library entry point. Implements the two exported C functions: `CreateEpFactories` (called by ORT after `dlopen`) initialises the ORT C++ API wrapper and allocates a `NeuronEpFactory`; `ReleaseEpFactory` deletes it. |
| `neuron_ep_factory.h/cc` | `NeuronEpFactory` — one instance per ORT environment. Fills the `OrtEpFactory` C vtable. Key responsibilities: advertises support for the first available CPU device (`GetSupportedDevices`), creates a `NeuronEp` per session (`CreateEp`), manages a reference-counted shared `CustomAllocator`, owns the singleton `NeuronDataTransfer`, and creates `NeuronStreamImpl` objects. |
| `neuron_ep.h/cc` | `NeuronEp` — one instance per ORT session. Fills the `OrtEp` C vtable. `GetCapability` walks the graph and claims any `Add` node whose two inputs are static, equal-shaped float32 tensors. `Compile` saves constant initializers then creates an `AddKernel` for the fused node and returns a `NeuronNodeComputeInfo` so ORT knows how to call it at inference time. `AddKernel::Compute` does the actual element-wise addition and handles the three cases: both inputs live, one input is a saved constant, or both are constants. |
| `neuron_ep_allocator.h` | `CustomAllocator` — thin `malloc`/`free` wrapper implementing `OrtAllocator`. Tracks allocation count and peak size for debugging via `GetStats`. |
| `neuron_ep_data_transfer.h/cc` | `NeuronDataTransfer` — implements `OrtDataTransferImpl`. `CanCopy` returns true for device↔device and device↔CPU transfers. `CopyTensors` uses `memcpy` (a real GPU EP would call `cudaMemcpyAsync` here). The object is factory-owned and shared across sessions; its `Release` is a no-op. |
| `neuron_ep_stream_support.h/cc` | `NeuronStreamImpl` and `NeuronNotificationImpl` — stub implementations of `OrtSyncStreamImpl` / `OrtSyncNotificationImpl`. All operations are no-ops because this is a CPU EP. A GPU EP would hold a `cudaStream_t` / `cudaEvent_t` here and call the corresponding CUDA/ACL APIs. |
| `neuron_ep_utils.h` | Shared utilities: error-propagation macros (`RETURN_IF_ERROR`, `RETURN_IF`), the `NEURON_LOG` logging macro, the `ApiPtrs` bundle struct (groups `OrtApi`, `OrtEpApi`, `OrtModelEditorApi`), the `FloatInitializer` struct for saved weights, and graph-inspection helpers (`IsFloatTensor`, `GetTensorShape`, `AreShapesStaticAndEqual`). |

### Python scripts

| File | What it does |
|---|---|
| `create_test_model.py` | Generates a minimal ONNX model with a single `Add` node. `--shape H W` sets the tensor shape. `--with-constant` produces a second variant where `B` is a constant initializer, exercising the `drop_constant_initializers` path. |
| `test_neuron_ep.py` | End-to-end test. Registers the `.so`, queries `OrtEpDevice` list, builds `SessionOptions` pointing at `NeuronDummyEP`, loads the model (triggering `GetCapability` + `Compile`), runs inference, and asserts the output matches `A + B`. Works with both model variants produced by `create_test_model.py`. |

---

## Quick start

```bash
# 1. Build
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/ort-install -DGSL_INCLUDE_DIR=/path/to/gsl/include
make -j$(nproc)

# 2. Generate test model
python3 create_test_model.py --shape 3 4

# 3. Run test
python3 test_neuron_ep.py --so build/neuron_dummy_plugin_ep.so --model add_model.onnx

# Cross-compile for aarch64
cmake .. -DCMAKE_TOOLCHAIN_FILE=../aarch64_toolchain.cmake \
         -DONNXRUNTIME_ROOT=/path/to/ort-aarch64-install \
         -DGSL_INCLUDE_DIR=/path/to/gsl/include
```
