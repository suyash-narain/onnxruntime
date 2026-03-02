// neuron_ep_utils.h
// Standalone utility macros and helpers for NeuronDummyPluginEP.
// Based on onnxruntime/test/autoep/library/plugin_ep_utils.h

#pragma once

#include <gsl/span>
#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// ---------------------------------------------------------------------------
// Error-handling macros
// ---------------------------------------------------------------------------

// Propagate a non-OK OrtStatus* returned by `fn` upward.
#define RETURN_IF_ERROR(fn)     \
  do {                          \
    Ort::Status _s{(fn)};       \
    if (!_s.IsOK()) {           \
      return _s.release();      \
    }                           \
  } while (0)

// Return an ORT_EP_FAIL status if `cond` is true.
#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

// Throw std::runtime_error if `condition` is false. Used in internal asserts.
#define EP_ENFORCE(condition, ...)                        \
  do {                                                    \
    if (!(condition)) {                                   \
      std::ostringstream oss;                             \
      oss << "EP_ENFORCE failed: " << #condition << " "; \
      oss << __VA_ARGS__;                                 \
      throw std::runtime_error(oss.str());                \
    }                                                     \
  } while (false)

// Take ownership of an OrtStatus* and silently discard it.
#define IGNORE_ORTSTATUS(expr)       \
  do {                               \
    OrtStatus* _s = (expr);          \
    Ort::Status _ignored{_s};        \
  } while (false)

// ---------------------------------------------------------------------------
// Logging macros (require members `api_` and `logger_` in scope)
// ---------------------------------------------------------------------------

#ifdef _WIN32
#define EP_WSTR(x) L##x
#define EP_FILE EP_WSTR(__FILE__)
#else
#define EP_FILE __FILE__
#endif

#define NEURON_LOG(level, ...)                                                       \
  do {                                                                               \
    std::ostringstream _ss;                                                          \
    _ss << __VA_ARGS__;                                                              \
    IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(&logger_,                             \
                                               ORT_LOGGING_LEVEL_##level,            \
                                               _ss.str().c_str(),                    \
                                               EP_FILE, __LINE__, __FUNCTION__));    \
  } while (false)

// ---------------------------------------------------------------------------
// Core structs shared across all EP files
// ---------------------------------------------------------------------------

// Bundles the three ORT API pointers every file needs.
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// Holds a saved constant float initializer (weight).
struct FloatInitializer {
  std::vector<int64_t> shape;
  std::vector<float>   data;
};

// ---------------------------------------------------------------------------
// Session-config helper
// ---------------------------------------------------------------------------

inline OrtStatus* GetSessionConfigEntryOrDefault(const OrtSessionOptions& session_options,
                                                  const char* config_key,
                                                  const std::string& default_val,
                                                  std::string& config_val) {
  try {
    Ort::ConstSessionOptions opts{&session_options};
    config_val = opts.GetConfigEntryOrDefault(config_key, default_val);
  } catch (const Ort::Exception& ex) {
    Ort::Status s(ex);
    return s.release();
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Graph-inspection helpers
// ---------------------------------------------------------------------------

// Sets `result` to true if `value_info` is a float tensor.
inline void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result) {
  result = false;
  auto type_info  = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) return;
  auto ts = type_info.GetTensorTypeAndShapeInfo();
  result  = (ts.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

// Returns the shape of a tensor ValueInfo, or nullopt if not a tensor.
inline std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info) {
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) return std::nullopt;
  return type_info.GetTensorTypeAndShapeInfo().GetShape();
}

// Returns true if both shapes have no dynamic dims and are identical.
inline bool AreShapesStaticAndEqual(const std::vector<int64_t>& s0,
                                     const std::vector<int64_t>& s1) {
  for (int64_t d : s0) if (d < 0) return false;
  for (int64_t d : s1) if (d < 0) return false;
  return s0 == s1;
}
