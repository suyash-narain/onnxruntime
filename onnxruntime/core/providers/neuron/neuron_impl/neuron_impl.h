// neuron_impl.h
// Dummy Add-op implementation of the NeuronWrapper_* interface.
//
// This is the "neuron_impl" — the accelerator-side code that the EP shell
// (onnxruntime/core/providers/neuron) calls through the interface declared in
// neuron_execution_provider_wrapper.h.
//
// For a real Neuron/NPU EP, replace AddKernel::Compute with actual SDK calls.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// ---------------------------------------------------------------------------
// AddKernel
// Executes element-wise float32 addition at inference time.
// ---------------------------------------------------------------------------
struct AddKernel {
  // ort_api is needed to create OrtStatus on error.
  explicit AddKernel(const OrtApi& ort_api) : ort_api(ort_api) {}

  OrtStatus* Compute(OrtKernelContext* ctx);

  const OrtApi& ort_api;
};

// ---------------------------------------------------------------------------
// NodeComputeInfoBase
// Virtual base so ReleaseNodeComputeInfos can polymorphically delete
// any derived type without manual type dispatch.
// ---------------------------------------------------------------------------
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

// ---------------------------------------------------------------------------
// AddNodeComputeInfo
// The OrtNodeComputeInfo returned by NeuronWrapper_Compile for each Add node.
// ---------------------------------------------------------------------------
struct AddNodeComputeInfo : NodeComputeInfoBase {
  // Raw pointer — AddKernel is owned by NeuronWrapperState.
  AddKernel* kernel{nullptr};

  explicit AddNodeComputeInfo(AddKernel* k);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                  OrtNodeComputeContext* ctx,
                                                  void** state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                              void* state,
                                              OrtKernelContext* kernel_ctx);
  static void ORT_API_CALL       ReleaseStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   void* state);
};

// ---------------------------------------------------------------------------
// NeuronWrapperState
// Per-session state created by NeuronWrapper_Create.
// ---------------------------------------------------------------------------
struct NeuronWrapperState {
  const OrtApi*            ort_api{nullptr};
  const OrtEpApi*          ep_api{nullptr};
  const OrtModelEditorApi* model_editor_api{nullptr};

  // One AddKernel per compiled fused node, keyed by ORT-assigned fused node name.
  std::unordered_map<std::string, std::unique_ptr<AddKernel>> add_kernels;
};
