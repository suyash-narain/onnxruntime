// neuron_impl.cc
// Implements the NeuronWrapper_* C interface declared in
// neuron_execution_provider_wrapper.h.
//
// This dummy implementation supports float32 Add nodes only.
// For a real Neuron/NPU EP replace AddKernel::Compute with SDK compilation
// and execution calls.

#include "neuron_impl.h"
#include "../neuron_execution_provider_wrapper.h"

#include <cassert>

// ============================================================================
// Internal helpers
// ============================================================================

static bool IsFloatTensor(const OrtValueInfo* vi) {
  Ort::ConstValueInfo info{vi};
  auto type_info = info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) return false;
  return type_info.GetTensorTypeAndShapeInfo().GetElementType() ==
         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// Returns the shape of the tensor, or an empty vector if any dim is dynamic.
static std::vector<int64_t> StaticShape(const OrtValueInfo* vi) {
  auto shape = Ort::ConstValueInfo{vi}.TypeInfo()
                   .GetTensorTypeAndShapeInfo()
                   .GetShape();
  for (int64_t d : shape)
    if (d < 0) return {};
  return shape;
}

// ============================================================================
// AddKernel::Compute
// Element-wise float32 Add.  Inputs must have identical static shapes.
// ============================================================================
OrtStatus* AddKernel::Compute(OrtKernelContext* ctx) {
  Ort::KernelContext kctx{ctx};

  Ort::ConstValue in0 = kctx.GetInput(0);
  Ort::ConstValue in1 = kctx.GetInput(1);

  auto tsi0 = in0.GetTensorTypeAndShapeInfo();
  auto tsi1 = in1.GetTensorTypeAndShapeInfo();

  if (tsi0.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      tsi1.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "NeuronImpl Add: expected float32 inputs");

  auto shape = tsi0.GetShape();
  if (shape != tsi1.GetShape())
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "NeuronImpl Add: input shapes must match");

  const float* a = in0.GetTensorData<float>();
  const float* b = in1.GetTensorData<float>();
  size_t        n = tsi0.GetElementCount();

  Ort::UnownedValue out = kctx.GetOutput(0, shape);
  float* y = out.GetTensorMutableData<float>();

  for (size_t i = 0; i < n; ++i)
    y[i] = a[i] + b[i];

  return nullptr;
}

// ============================================================================
// AddNodeComputeInfo
// ============================================================================
AddNodeComputeInfo::AddNodeComputeInfo(AddKernel* k) : kernel(k) {
  ort_version_supported = ORT_API_VERSION;
  CreateState  = CreateStateImpl;
  Compute      = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

/*static*/
OrtStatus* ORT_API_CALL AddNodeComputeInfo::CreateStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/,
    OrtNodeComputeContext* /*ctx*/,
    void** state) {
  // The kernel pointer lives in AddNodeComputeInfo and is accessed directly
  // in ComputeImpl via this_ptr, so compute_state is unused.
  *state = nullptr;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AddNodeComputeInfo::ComputeImpl(
    OrtNodeComputeInfo* this_ptr,
    void* /*state*/,
    OrtKernelContext* kernel_ctx) {
  return static_cast<AddNodeComputeInfo*>(this_ptr)->kernel->Compute(kernel_ctx);
}

/*static*/
void ORT_API_CALL AddNodeComputeInfo::ReleaseStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* /*state*/) {
  // Nothing to release — AddKernel is owned by NeuronWrapperState.
}

// ============================================================================
// NeuronWrapper_* implementations
// ============================================================================

extern "C" {

// ---------------------------------------------------------------------------
// NeuronWrapper_Create
// Allocate and initialise per-session state.
// ---------------------------------------------------------------------------
NeuronWrapperHandle NeuronWrapper_Create(
    const OrtApi*            ort_api,
    const OrtEpApi*          ep_api,
    const OrtModelEditorApi* model_editor_api,
    const OrtKeyValuePairs*  /*provider_options*/) {
  auto* state = new (std::nothrow) NeuronWrapperState{};
  if (!state) return nullptr;
  state->ort_api          = ort_api;
  state->ep_api           = ep_api;
  state->model_editor_api = model_editor_api;
  return state;
}

// ---------------------------------------------------------------------------
// NeuronWrapper_Destroy
// ---------------------------------------------------------------------------
void NeuronWrapper_Destroy(NeuronWrapperHandle handle) {
  delete static_cast<NeuronWrapperState*>(handle);
}

// ---------------------------------------------------------------------------
// NeuronWrapper_GetCapability
// Walk the graph; claim any float32 Add node whose two inputs have identical
// static shapes (no broadcasting, no dynamic dims).
// ---------------------------------------------------------------------------
OrtStatus* NeuronWrapper_GetCapability(
    NeuronWrapperHandle    /*handle*/,
    const OrtApi*          /*ort_api*/,
    const OrtEpApi*        ep_api,
    const OrtGraph*        graph,
    OrtEpGraphSupportInfo* support_info) {
  Ort::ConstGraph g{graph};
  std::vector<Ort::ConstNode> nodes = g.GetNodes();

  for (const auto& node : nodes) {
    std::string op  = node.GetOperatorType();
    std::string dom = node.GetDomain();

    // Only claim Add in the default ONNX domain.
    if (op != "Add") continue;
    if (!dom.empty() && dom != "ai.onnx") continue;

    auto inputs  = node.GetInputs();
    auto outputs = node.GetOutputs();

    if (inputs.size() != 2 || outputs.size() != 1) continue;

    // Require float32 on all inputs and the output.
    if (!IsFloatTensor(inputs[0]) ||
        !IsFloatTensor(inputs[1]) ||
        !IsFloatTensor(outputs[0])) continue;

    // Require same static shape on both inputs (no broadcast, no dynamic dims).
    auto sh0 = StaticShape(inputs[0]);
    auto sh1 = StaticShape(inputs[1]);
    if (sh0.empty() || sh0 != sh1) continue;

    // Register this single node as a fused subgraph.
    OrtNodeFusionOptions opts{};
    opts.ort_version_supported      = ORT_API_VERSION;
    opts.drop_constant_initializers = false;

    const OrtNode* raw_node = node;
    OrtStatus* st = ep_api->EpGraphSupportInfo_AddNodesToFuse(
        support_info, &raw_node, 1, &opts);
    if (st) return st;

    // Claim only one Add per graph for this dummy implementation.
    break;
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
// NeuronWrapper_Compile
// For each fused subgraph (containing a single Add node), create an AddKernel
// and return an AddNodeComputeInfo to ORT.
//
// For a real NPU EP this is where you'd invoke the SDK compiler and store
// the resulting binary in AddKernel for use at inference time.
// ---------------------------------------------------------------------------
OrtStatus* NeuronWrapper_Compile(
    NeuronWrapperHandle          handle,
    const OrtApi*                ort_api,
    const OrtEpApi*              ep_api,
    const OrtModelEditorApi*     /*model_editor_api*/,
    const OrtGraph* const*       graphs,
    const OrtNode*  const*       fused_nodes,
    size_t                       count,
    OrtNodeComputeInfo**         node_compute_infos,
    OrtNode**                    /*ep_context_nodes*/) {
  auto* state = static_cast<NeuronWrapperState*>(handle);

  for (size_t i = 0; i < count; ++i) {
    Ort::ConstGraph  g{graphs[i]};
    Ort::ConstNode   fused_node{fused_nodes[i]};
    std::string      fused_name = fused_node.GetName();

    // Validate: expect exactly one Add node in the fused subgraph.
    auto nodes = g.GetNodes();
    if (nodes.size() != 1 || std::string(nodes[0].GetOperatorType()) != "Add") {
      return ort_api->CreateStatus(ORT_EP_FAIL,
          "NeuronImpl: expected a single Add node in the fused subgraph");
    }

    // Create the AddKernel (in a real EP, SDK compilation happens here).
    auto kernel    = std::make_unique<AddKernel>(*ort_api);
    auto* raw_kern = kernel.get();
    state->add_kernels.emplace(fused_name, std::move(kernel));

    // Return the OrtNodeComputeInfo to ORT.
    auto nci = std::make_unique<AddNodeComputeInfo>(raw_kern);
    node_compute_infos[i] = nci.release();
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
// NeuronWrapper_ReleaseNodeComputeInfos
// ---------------------------------------------------------------------------
void NeuronWrapper_ReleaseNodeComputeInfos(
    NeuronWrapperHandle  /*handle*/,
    OrtNodeComputeInfo** node_compute_infos,
    size_t               count) {
  for (size_t i = 0; i < count; ++i)
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
}

// ---------------------------------------------------------------------------
// NeuronWrapper_GetPreferredLayout
// Return NHWC (1) — typical for mobile NPUs.
// ---------------------------------------------------------------------------
int NeuronWrapper_GetPreferredLayout(NeuronWrapperHandle /*handle*/) {
  return static_cast<int>(OrtEpDataLayout_NHWC);
}

// ---------------------------------------------------------------------------
// NeuronWrapper_ShouldSkipLayoutConversion
// Return 0 (allow conversion) for all ops.
// A real implementation would return 1 for ops like Softmax.
// ---------------------------------------------------------------------------
int NeuronWrapper_ShouldSkipLayoutConversion(
    NeuronWrapperHandle /*handle*/,
    const char*         /*domain*/,
    const char*         /*op_type*/,
    int                 /*target_layout*/) {
  return 0;
}

}  // extern "C"
