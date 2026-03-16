// neuron_ep.h
// NeuronEp: per-session execution provider.
// Delegates GetCapability and Compile to libonnxruntime_provider_neuron_wrapper.so
// via a stable C ABI defined in neuron_execution_provider_wrapper.h.

#pragma once

#include <memory>
#include <string>

#include "neuron_ep_utils.h"
#include "neuron_execution_provider_wrapper.h"

class NeuronEpFactory;  // forward declaration

// ---------------------------------------------------------------------------
// NeuronWrapperLib
// Manages dlopen of libonnxruntime_provider_neuron_wrapper.so and the resolved
// function pointers.  One instance is owned by each NeuronEp (per-session).
// ---------------------------------------------------------------------------
struct NeuronWrapperLib {
  using Fn_Create      = decltype(&NeuronWrapper_Create);
  using Fn_Destroy     = decltype(&NeuronWrapper_Destroy);
  using Fn_GetCap      = decltype(&NeuronWrapper_GetCapability);
  using Fn_Compile     = decltype(&NeuronWrapper_Compile);
  using Fn_Release     = decltype(&NeuronWrapper_ReleaseNodeComputeInfos);
  using Fn_Layout      = decltype(&NeuronWrapper_GetPreferredLayout);
  using Fn_SkipLayout  = decltype(&NeuronWrapper_ShouldSkipLayoutConversion);

  Fn_Create     Create{};
  Fn_Destroy    Destroy{};
  Fn_GetCap     GetCapability{};
  Fn_Compile    Compile{};
  Fn_Release    ReleaseNodeComputeInfos{};
  Fn_Layout     GetPreferredLayout{};
  Fn_SkipLayout ShouldSkipLayoutConversion{};

  void* lib_handle{nullptr};

  ~NeuronWrapperLib();

  // Load the shared library and resolve all required symbols.
  // Returns an empty string on success or an error message on failure.
  std::string Load(const char* lib_path);
};

// ---------------------------------------------------------------------------
// NeuronEp
// One instance per ORT session.  Inherits OrtEp (C vtable struct) and ApiPtrs.
// ---------------------------------------------------------------------------
class NeuronEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    // Path to the wrapper shared library.
    std::string wrapper_lib_path{"libonnxruntime_provider_neuron_wrapper.so"};

    // Provider options forwarded verbatim to NeuronWrapper_Create.
    // Pointer is NOT owned; it is owned by NeuronEpFactory.
    const OrtKeyValuePairs* provider_options{nullptr};
  };

  NeuronEp(NeuronEpFactory& factory,
           const std::string& ep_name,
           const Config& config,
           const OrtLogger& logger);

  ~NeuronEp();

 private:
  // ---- OrtEp vtable implementations (all static, ORT_API_CALL) ------------
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* ep,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* ep,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* ep,
      OrtNodeComputeInfo** infos,
      size_t num) noexcept;

  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(
      OrtEp* ep,
      OrtEpDataLayout* preferred_data_layout) noexcept;

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(
      OrtEp* ep,
      const char* domain,
      const char* op_type,
      OrtEpDataLayout target_data_layout,
      int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEp* ep,
      const OrtMemoryInfo* memory_info,
      OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* ep,
      const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  // ---- Per-session state --------------------------------------------------
  NeuronEpFactory&    factory_;
  std::string         name_;
  Config              config_;
  const OrtLogger&    logger_;

  NeuronWrapperLib    wrapper_lib_;
  NeuronWrapperHandle neuron_handle_{nullptr};
};
