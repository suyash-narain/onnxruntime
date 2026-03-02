// neuron_dummy_plugin_ep.cc
// DLL / shared-library entry point.
//
// ORT calls CreateEpFactories() immediately after dlopen().
// ReleaseEpFactory() is called when the ORT environment is destroyed.
//
// Only these two C symbols are exported (see .lds / .def files).

#define ORT_API_MANUAL_INIT   // suppress automatic global init inside a DLL
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "neuron_ep_factory.h"

// macOS needs an explicit visibility attribute; Linux/Windows rely on .lds/.def.
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {

// ---------------------------------------------------------------------------
// CreateEpFactories
// Called by ORT once after loading this shared library.
//   registration_name – the name the user passed to
//                       register_execution_provider_library().
//   ort_api_base      – use this to obtain the versioned ORT C API.
//   default_logger    – process-level logger.
//   factories / max_factories / num_factories – output array.
// ---------------------------------------------------------------------------
EXPORT_SYMBOL OrtStatus* CreateEpFactories(
    const char*          registration_name,
    const OrtApiBase*    ort_api_base,
    const OrtLogger*     default_logger,
    OrtEpFactory**       factories,
    size_t               max_factories,
    size_t*              num_factories) {

  // 1. Obtain versioned APIs.
  const OrtApi*           ort_api          = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi*         ep_api           = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // 2. Manually initialise the C++ wrapper (required in DLLs).
  Ort::InitApi(ort_api);

  // 3. Sanity-check output buffer size.
  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                  "Not enough space to return EP factory. Need at least 1.");
  }

  // 4. Create the factory.  The name comes from the caller so users can
  //    register the same .so under any name they like.
  auto factory = std::make_unique<NeuronEpFactory>(
      registration_name,
      ApiPtrs{*ort_api, *ep_api, *model_editor_api},
      *default_logger);

  factories[0]  = factory.release();
  *num_factories = 1;
  return nullptr;
}

// ---------------------------------------------------------------------------
// ReleaseEpFactory
// Called by ORT when the environment is shut down.
// ---------------------------------------------------------------------------
EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<NeuronEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
