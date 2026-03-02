// neuron_ep.h
// Declares AddKernel (element-wise float addition) and NeuronEp (OrtEp subclass).
// NeuronEp is the per-session execution provider object.

#pragma once

#include <unordered_map>
#include <memory>
#include <string>

#include <gsl/span>
#include "neuron_ep_utils.h"

class NeuronEpFactory;  // forward declare

// ---------------------------------------------------------------------------
// AddKernel
// Stores per-node state for a compiled Add node.  Compute() is called at
// every inference run.
// ---------------------------------------------------------------------------
struct AddKernel {
  AddKernel(const OrtApi& ort_api,
            const OrtLogger& logger,
            const std::unordered_map<std::string, FloatInitializer>& float_initializers,
            std::string input0_name,
            std::string input1_name)
      : ort_api(ort_api),
        logger(logger),
        float_initializers(float_initializers),
        input0_name(std::move(input0_name)),
        input1_name(std::move(input1_name)) {}

  // Returns saved initializer by name, or nullptr if not found.
  const FloatInitializer* TryGetSavedInitializer(const std::string& name) const;

  // Reads one input from OrtKernelContext by index.
  void GetInputDataAndShape(Ort::KernelContext ctx, size_t idx,
                            /*out*/ gsl::span<const float>& data,
                            /*out*/ std::vector<int64_t>& shape) const;

  // Main compute: reads inputs, writes output = A + B (element-wise).
  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi&                                                ort_api;
  const OrtLogger&                                             logger;
  const std::unordered_map<std::string, FloatInitializer>&     float_initializers;
  std::string input0_name;
  std::string input1_name;
};

// ---------------------------------------------------------------------------
// NeuronEp
// One instance per ORT session.  Inherits OrtEp (C vtable struct) and ApiPtrs.
// ---------------------------------------------------------------------------
class NeuronEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    // Add EP-specific session config options here.
    // e.g.: bool use_fp16 = false;
  };

  NeuronEp(NeuronEpFactory& factory,
           const std::string& name,
           const Config& config,
           const OrtLogger& logger);

  ~NeuronEp();

  // Accessor used by NeuronNodeComputeInfo to look up kernels at run-time.
  std::unordered_map<std::string, std::unique_ptr<AddKernel>>& AddKernels() {
    return add_kernels_;
  }

 private:
  // ---- OrtEp vtable implementations (all static, ORT_API_CALL) ------------
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEp* ep, const OrtMemoryInfo* memory_info,
      OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* ep, const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* ep, const OrtGraph* graph,
      OrtEpGraphSupportInfo* support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* ep,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* ep, OrtNodeComputeInfo** infos, size_t num) noexcept;

  // Iterates graph initializers and saves constant float tensors.
  OrtStatus* SaveConstantInitializers(const OrtGraph* graph);

  // ---- Per-session state --------------------------------------------------
  NeuronEpFactory& factory_;
  std::string      name_;
  Config           config_{};
  const OrtLogger& logger_;

  // Map: fused-node-name → compiled AddKernel
  std::unordered_map<std::string, std::unique_ptr<AddKernel>> add_kernels_;

  // Saved weights (constant initializers dropped from ORT's graph after Compile)
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
};
