// neuron_ep.cc
// Implementation of AddKernel and NeuronEp.

#include "neuron_ep.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "neuron_ep_factory.h"
#include "neuron_ep_stream_support.h"

// ===========================================================================
// AddKernel
// ===========================================================================

const FloatInitializer* AddKernel::TryGetSavedInitializer(const std::string& name) const {
  auto it = float_initializers.find(name);
  return it != float_initializers.end() ? &it->second : nullptr;
}

void AddKernel::GetInputDataAndShape(Ort::KernelContext ctx, size_t idx,
                                     gsl::span<const float>& data,
                                     std::vector<int64_t>& shape) const {
  Ort::ConstValue v = ctx.GetInput(idx);
  auto ts = v.GetTensorTypeAndShapeInfo();
  if (ts.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw Ort::Exception("NeuronEP: expected float32 input", ORT_EP_FAIL);
  data  = gsl::span<const float>(v.GetTensorData<float>(), ts.GetElementCount());
  shape = ts.GetShape();
}

OrtStatus* AddKernel::Compute(OrtKernelContext* kernel_ctx) {
  RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger,
                                            ORT_LOGGING_LEVEL_INFO,
                                            "AddKernel::Compute",
                                            ORT_FILE, __LINE__, __FUNCTION__));
  Ort::KernelContext ctx(kernel_ctx);
  try {
    gsl::span<const float> input0, input1;
    std::vector<int64_t>   shape0, shape1;

    size_t num_inputs = ctx.GetInputCount();

    if (num_inputs == 2) {
      // Both are live tensors from ORT.
      GetInputDataAndShape(ctx, 0, input0, shape0);
      GetInputDataAndShape(ctx, 1, input1, shape1);
    } else if (num_inputs == 1) {
      // One input was a constant initializer we saved during Compile().
      // ORT dropped it because we set drop_constant_initializers = true.
      if (const FloatInitializer* c0 = TryGetSavedInitializer(input0_name)) {
        GetInputDataAndShape(ctx, 0, input1, shape1);
        input0 = gsl::span<const float>(c0->data);
        shape0 = c0->shape;
      } else if (const FloatInitializer* c1 = TryGetSavedInitializer(input1_name)) {
        GetInputDataAndShape(ctx, 0, input0, shape0);
        input1 = gsl::span<const float>(c1->data);
        shape1 = c1->shape;
      } else {
        return ort_api.CreateStatus(ORT_EP_FAIL, "AddKernel: single input but no saved initializer found");
      }
    } else {
      // Both are constants (constant-folding disabled).
      const FloatInitializer* c0 = TryGetSavedInitializer(input0_name);
      const FloatInitializer* c1 = TryGetSavedInitializer(input1_name);
      RETURN_IF(!c0 || !c1, ort_api, "AddKernel: 0 live inputs but saved initializers missing");
      input0 = gsl::span<const float>(c0->data); shape0 = c0->shape;
      input1 = gsl::span<const float>(c1->data); shape1 = c1->shape;
    }

    if (shape0 != shape1)
      throw Ort::Exception("NeuronEP AddKernel: shape mismatch", ORT_INVALID_ARGUMENT);
    if (ctx.GetOutputCount() != 1)
      throw Ort::Exception("NeuronEP AddKernel: expected 1 output", ORT_INVALID_ARGUMENT);

    auto output = ctx.GetOutput(0, shape0);
    float* out_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < input0.size(); ++i)
      out_data[i] = input0[i] + input1[i];   // ← the actual Add computation

  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
  return nullptr;
}

// ===========================================================================
// NeuronNodeComputeInfo
// Returned by Compile() for each fused node.  ORT calls these three functions
// at session-run time: CreateState → Compute → ReleaseState.
// ===========================================================================

// Polymorphic base so ReleaseNodeComputeInfosImpl can delete via base pointer.
struct NeuronNodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NeuronNodeComputeInfoBase() = default;
};

struct NeuronNodeComputeInfo : NeuronNodeComputeInfoBase {
  explicit NeuronNodeComputeInfo(NeuronEp& ep) : ep(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState  = CreateStateImpl;
    Compute      = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  // Called once at session-run start: look up the right AddKernel.
  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                  OrtNodeComputeContext* ctx,
                                                  void** compute_state) {
    auto* nci = static_cast<NeuronNodeComputeInfo*>(this_ptr);
    NeuronEp& ep = nci->ep;

    std::string node_name = ep.ep_api.NodeComputeContext_NodeName(ctx);
    auto it = ep.AddKernels().find(node_name);
    if (it == ep.AddKernels().end()) {
      std::string msg = "NeuronEP: no AddKernel for fused node '" + node_name + "'";
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  // Called every inference run.
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                              void* compute_state,
                                              OrtKernelContext* kernel_ctx) {
    return reinterpret_cast<AddKernel*>(compute_state)->Compute(kernel_ctx);
  }

  // Called at session destruction (nothing to do; kernel owned by NeuronEp).
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                             void* /*compute_state*/) {}

  NeuronEp& ep;
};

// ===========================================================================
// NeuronEp
// ===========================================================================

NeuronEp::NeuronEp(NeuronEpFactory& factory,
                   const std::string& name,
                   const Config& config,
                   const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      logger_{logger} {
  // Tell ORT which API version we were compiled with.
  ort_version_supported = ORT_API_VERSION;

  // Populate the C vtable (function pointers on OrtEp).
  GetName                      = GetNameImpl;
  GetCapability                = GetCapabilityImpl;
  Compile                      = CompileImpl;
  ReleaseNodeComputeInfos      = ReleaseNodeComputeInfosImpl;
  CreateAllocator              = CreateAllocatorImpl;             // optional
  CreateSyncStreamForDevice    = CreateSyncStreamForDeviceImpl;   // optional

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
      &logger_, ORT_LOGGING_LEVEL_INFO,
      ("NeuronEp created: " + name_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));
}

NeuronEp::~NeuronEp() = default;

// ---- GetName ---------------------------------------------------------------
/*static*/
const char* ORT_API_CALL NeuronEp::GetNameImpl(const OrtEp* ep) noexcept {
  return static_cast<const NeuronEp*>(ep)->name_.c_str();
}

// ---- SaveConstantInitializers ----------------------------------------------
OrtStatus* NeuronEp::SaveConstantInitializers(const OrtGraph* ort_graph) {
  Ort::ConstGraph graph{ort_graph};
  try {
    for (const auto& init : graph.GetInitializers()) {
      if (!init.IsConstantInitializer()) continue;

      std::string name = init.GetName();
      Ort::ConstValue val;
      {
        auto st = init.GetInitializer(val);
        if (!st.IsOK()) return st.release();
      }

      auto ts = val.GetTensorTypeAndShapeInfo();
      if (ts.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        continue;  // only handle float weights in this example

      const float* data = val.GetTensorData<float>();
      std::vector<int64_t> dims = ts.GetShape();
      size_t n = ts.GetElementCount();

      float_initializers_.emplace(name, FloatInitializer{dims, {data, data + n}});
    }
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
  return nullptr;
}

// ---- GetCapability ---------------------------------------------------------
// ORT calls this during graph partitioning.  We walk the graph looking for
// Add nodes with float inputs of equal static shape.
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::GetCapabilityImpl(OrtEp* ep_ptr,
                                                     const OrtGraph* ort_graph,
                                                     OrtEpGraphSupportInfo* support_info) noexcept {
  try {
    NeuronEp* ep    = static_cast<NeuronEp*>(ep_ptr);
    Ort::ConstGraph graph{ort_graph};
    auto nodes = graph.GetNodes();
    if (nodes.empty()) return nullptr;

    for (const auto& node : nodes) {
      if (node.GetOperatorType() != std::string("Add")) continue;
      if (node.GetDomain()       != std::string(""))    continue;  // standard ONNX domain

      auto inputs  = node.GetInputs();
      auto outputs = node.GetOutputs();
      if (inputs.size() != 2 || outputs.size() != 1) continue;

      // Both inputs and the output must be float.
      bool f0, f1, fo;
      IsFloatTensor(inputs[0], f0);
      IsFloatTensor(inputs[1], f1);
      IsFloatTensor(outputs[0], fo);
      if (!f0 || !f1 || !fo) continue;

      // Require static, equal shapes (no broadcasting in this example).
      auto s0 = GetTensorShape(inputs[0]);
      auto s1 = GetTensorShape(inputs[1]);
      if (!s0 || !s1) continue;
      if (!AreShapesStaticAndEqual(*s0, *s1)) continue;

      // Tell ORT: fuse this node into a subgraph for us to compile.
      // drop_constant_initializers = true → we will save weights in Compile()
      //                              and ORT need not pass them at inference time.
      OrtNodeFusionOptions opts{};
      opts.ort_version_supported      = ORT_API_VERSION;
      opts.drop_constant_initializers = true;

      const OrtNode* raw = node;
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          support_info, &raw, 1, &opts));

      break;  // This EP compiles one Add at a time (same pattern as example_plugin_ep).
    }
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
  return nullptr;
}

// ---- Compile ---------------------------------------------------------------
// ORT calls this once per fused subgraph.  We create an AddKernel and return
// a NeuronNodeComputeInfo so ORT knows how to run inference.
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::CompileImpl(OrtEp* ep_ptr,
                                               const OrtGraph** ort_graphs,
                                               const OrtNode** fused_nodes,
                                               size_t count,
                                               OrtNodeComputeInfo** node_compute_infos,
                                               OrtNode** /*ep_context_nodes*/) noexcept {
  try {
    if (count != 1) {
      return Ort::Status("NeuronEP: expected to compile exactly one graph", ORT_EP_FAIL).release();
    }

    NeuronEp* ep = static_cast<NeuronEp*>(ep_ptr);

    // Step 1: save any constant initializers before ORT releases them.
    RETURN_IF_ERROR(ep->SaveConstantInitializers(ort_graphs[0]));

    // Step 2: get the single node inside this subgraph.
    Ort::ConstGraph graph{ort_graphs[0]};
    auto sub_nodes = graph.GetNodes();
    if (sub_nodes.size() != 1)
      return Ort::Status("NeuronEP: expected subgraph with exactly one node", ORT_EP_FAIL).release();

    if (sub_nodes[0].GetOperatorType() != std::string("Add"))
      return Ort::Status("NeuronEP: expected Add node in subgraph", ORT_EP_FAIL).release();

    // Step 3: validate fused node is assigned to this EP.
    Ort::ConstNode fused{fused_nodes[0]};
    if (std::string(fused.GetEpName()) != ep->name_)
      return Ort::Status("NeuronEP: fused node is not assigned to this EP", ORT_EP_FAIL).release();

    std::string fused_name = fused.GetName();

    // Step 4: get Add's input names (for the saved-initializer lookup at runtime).
    auto add_inputs = sub_nodes[0].GetInputs();
    if (add_inputs.size() != 2)
      return Ort::Status("NeuronEP: Add node must have exactly 2 inputs", ORT_EP_FAIL).release();

    std::string input0_name = add_inputs[0].GetName();
    std::string input1_name = add_inputs[1].GetName();

    // Step 5: create kernel and wrap it in NeuronNodeComputeInfo.
    ep->add_kernels_.emplace(fused_name,
                             std::make_unique<AddKernel>(ep->ort_api,
                                                         ep->logger_,
                                                         ep->float_initializers_,
                                                         input0_name,
                                                         input1_name));

    auto nci = std::make_unique<NeuronNodeComputeInfo>(*ep);
    node_compute_infos[0] = nci.release();

  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
  return nullptr;
}

// ---- ReleaseNodeComputeInfos -----------------------------------------------
/*static*/
void ORT_API_CALL NeuronEp::ReleaseNodeComputeInfosImpl(OrtEp* /*ep*/,
                                                         OrtNodeComputeInfo** infos,
                                                         size_t num) noexcept {
  for (size_t i = 0; i < num; ++i)
    delete static_cast<NeuronNodeComputeInfoBase*>(infos[i]);
}

// ---- CreateAllocator (per-session, optional) --------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::CreateAllocatorImpl(OrtEp* ep_ptr,
                                                       const OrtMemoryInfo* memory_info,
                                                       OrtAllocator** allocator) noexcept {
  // Delegate to the factory's shared allocator logic.
  NeuronEp* ep = static_cast<NeuronEp*>(ep_ptr);
  return ep->factory_.CreateAllocator(&ep->factory_, memory_info, nullptr, allocator);
}

// ---- CreateSyncStreamForDevice (optional) ----------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::CreateSyncStreamForDeviceImpl(
    OrtEp* ep_ptr,
    const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
  NeuronEp* ep = static_cast<NeuronEp*>(ep_ptr);

  // Only create streams for the default device memory type.
  auto mem_type = ep->ep_api.MemoryDevice_GetMemoryType(memory_device);
  if (mem_type != OrtDeviceMemoryType_DEFAULT) {
    std::string err = "NeuronEP: stream requested for unsupported memory type: "
                    + std::to_string(mem_type);
    return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, err.c_str());
  }

  auto s = std::make_unique<NeuronStreamImpl>(ep->factory_, nullptr);
  *stream = s.release();
  return nullptr;
}
