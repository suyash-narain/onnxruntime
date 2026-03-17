// neuron_ep_factory.cc

#include "neuron_ep_factory.h"

#include <cassert>
#include <cstring>

#include "neuron_ep.h"
#include "neuron_ep_stream_support.h"

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
NeuronEpFactory::NeuronEpFactory(const char* ep_name, ApiPtrs apis,
                                 const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_{default_logger},
      ep_name_{ep_name},
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr} {
  ort_version_supported = NEURON_EP_ORT_API_VERSION;

  // Populate the C vtable.
  GetName                                 = GetNameImpl;
  GetVendor                               = GetVendorImpl;
  GetVendorId                             = GetVendorIdImpl;
  GetVersion                              = GetVersionImpl;
  GetSupportedDevices                     = GetSupportedDevicesImpl;
  CreateEp                                = CreateEpImpl;
  ReleaseEp                               = ReleaseEpImpl;
  CreateAllocator                         = CreateAllocatorImpl;
  ReleaseAllocator                        = ReleaseAllocatorImpl;
  CreateDataTransfer                      = CreateDataTransferImpl;
  IsStreamAware                           = IsStreamAwareImpl;
  CreateSyncStreamForDevice               = CreateSyncStreamForDeviceImpl;
  GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

  // Neuron EP uses CPU-accessible memory.
  // The MediaTek NPU reads/writes via the same CPU address space; no separate
  // device memory pool is needed at the ORT level.
  default_memory_info_ = Ort::MemoryInfo{
      "Neuron",
      OrtMemoryInfoDeviceType_CPU,
      /*vendor*/    vendor_id_,
      /*device_id*/ 0,
      OrtDeviceMemoryType_DEFAULT,
      /*alignment*/ 0,
      OrtAllocatorType::OrtDeviceAllocator};

  readonly_memory_info_ = Ort::MemoryInfo{
      "Neuron readonly",
      OrtMemoryInfoDeviceType_CPU,
      /*vendor*/    vendor_id_,
      /*device_id*/ 0,
      OrtDeviceMemoryType_DEFAULT,
      /*alignment*/ 0,
      OrtAllocatorType::OrtReadOnlyAllocator};

  // Create shared data-transfer (memcpy; works because memory is CPU-accessible).
  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_ = std::make_unique<NeuronDataTransfer>(apis, device);

  // Allocate an empty provider_options KV store; flags are merged in
  // GetSupportedDevices and read back in CreateEp.
  ort_api.CreateKeyValuePairs(&provider_options_);
}

NeuronEpFactory::~NeuronEpFactory() {
  if (provider_options_) {
    ort_api.ReleaseKeyValuePairs(provider_options_);
    provider_options_ = nullptr;
  }
}

// ---- Identity getters -------------------------------------------------------
/*static*/ const char* ORT_API_CALL NeuronEpFactory::GetNameImpl(const OrtEpFactory* f) noexcept {
  return static_cast<const NeuronEpFactory*>(f)->ep_name_.c_str();
}
/*static*/ const char* ORT_API_CALL NeuronEpFactory::GetVendorImpl(const OrtEpFactory* f) noexcept {
  return static_cast<const NeuronEpFactory*>(f)->vendor_.c_str();
}
/*static*/ uint32_t ORT_API_CALL NeuronEpFactory::GetVendorIdImpl(const OrtEpFactory* f) noexcept {
  return static_cast<const NeuronEpFactory*>(f)->vendor_id_;
}
/*static*/ const char* ORT_API_CALL NeuronEpFactory::GetVersionImpl(const OrtEpFactory* f) noexcept {
  return static_cast<const NeuronEpFactory*>(f)->ep_version_.c_str();
}

// ---- GetSupportedDevices ----------------------------------------------------
// Neuron EP targets MediaTek NPU hardware.  ORT enumerates hardware devices;
// we claim the first CPU device (Neuron SDK manages the NPU internally and
// presents a CPU-memory interface to ORT, same as NNAPI).
//
// The ep_metadata KV pairs are visible to users querying the registered EP
// and are forwarded back to CreateEp() so NeuronEp can read them.
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* f_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  size_t& count = *p_num_ep_devices;

  for (size_t i = 0; i < num_devices && count < max_ep_devices; ++i) {
    // Claim CPU hardware.  Neuron SDK acceleration is transparent to ORT;
    // from ORT's perspective the EP presents a CPU-memory interface.
    if (f.ort_api.HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) continue;

    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options  = nullptr;
    f.ort_api.CreateKeyValuePairs(&ep_metadata);
    f.ort_api.CreateKeyValuePairs(&ep_options);

    // Metadata visible to users querying the EP device.
    f.ort_api.AddKeyValuePair(ep_metadata, "version",  f.ep_version_.c_str());
    f.ort_api.AddKeyValuePair(ep_metadata, "vendor",   f.vendor_.c_str());
    f.ort_api.AddKeyValuePair(ep_metadata, "ep_name",  f.ep_name_.c_str());

    // Default option values (users can override via session options).
    f.ort_api.AddKeyValuePair(ep_options, "NEURON_FLAG_USE_FP16",  "0");
    f.ort_api.AddKeyValuePair(ep_options, "NEURON_FLAG_USE_NCHW",  "0");
    f.ort_api.AddKeyValuePair(ep_options, "NEURON_FLAG_CPU_DISABLED", "0");
    f.ort_api.AddKeyValuePair(ep_options, "NEURON_FLAG_CPU_ONLY",  "0");
    f.ort_api.AddKeyValuePair(ep_options, "NEURON_FLAG_MDLA_ONLY", "0");

    OrtEpDevice* ep_device = nullptr;
    OrtStatus* st = f.ep_api.CreateEpDevice(&f, devices[i], ep_metadata, ep_options, &ep_device);
    f.ort_api.ReleaseKeyValuePairs(ep_metadata);
    f.ort_api.ReleaseKeyValuePairs(ep_options);
    if (st) return st;

    // Register the memory types this EP uses so ORT can set up allocators.
    RETURN_IF_ERROR(f.ep_api.EpDevice_AddAllocatorInfo(ep_device, f.default_memory_info_));
    RETURN_IF_ERROR(f.ep_api.EpDevice_AddAllocatorInfo(ep_device, f.readonly_memory_info_));

    ep_devices[count++] = ep_device;
  }
  return nullptr;
}

// ---- CreateEp ---------------------------------------------------------------
// Called once per ORT session.  Reads Neuron flags from ep_metadata and
// session options and creates a NeuronEp instance.
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::CreateEpImpl(
    OrtEpFactory* f_ptr,
    const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* ep_metadata,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  *ep = nullptr;

  if (num_devices != 1)
    return f.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "NeuronEP: expected exactly one device.");

  // Build the provider options KV store that NeuronEp passes to the wrapper.
  // Start from the default options set in GetSupportedDevices, then apply any
  // user overrides read from the session config.
  OrtKeyValuePairs* options = nullptr;
  f.ort_api.CreateKeyValuePairs(&options);

  // Merge ep_metadata[0] flags (set by user via add_provider_option_for_ep_device).
  if (ep_metadata && ep_metadata[0]) {
    // Copy relevant Neuron flags from ep_metadata into options.
    static const char* kNeuronFlags[] = {
        "NEURON_FLAG_USE_FP16",
        "NEURON_FLAG_USE_NCHW",
        "NEURON_FLAG_CPU_DISABLED",
        "NEURON_FLAG_CPU_ONLY",
        "NEURON_FLAG_MDLA_ONLY",
        "NEURON_FLAG_MIN_GROUP_SIZE",
        "NEURON_FLAG_OPTIMIZATION_STRING",
        "NEURON_FLAG_FORCE_SKIP_OPS_STRING",
    };
    for (const char* key : kNeuronFlags) {
      const char* val = f.ort_api.GetKeyValue(ep_metadata[0], key);
      if (val) f.ort_api.AddKeyValuePair(options, key, val);
    }
  }

  // Also apply session config overrides (ep.neuron.<flag> = "1").
  if (session_options) {
    std::string config_val;
    static const struct { const char* cfg; const char* opt; } kConfigMap[] = {
        {"ep.neuron.use_fp16",             "NEURON_FLAG_USE_FP16"},
        {"ep.neuron.use_nchw",             "NEURON_FLAG_USE_NCHW"},
        {"ep.neuron.cpu_disabled",         "NEURON_FLAG_CPU_DISABLED"},
        {"ep.neuron.cpu_only",             "NEURON_FLAG_CPU_ONLY"},
        {"ep.neuron.mdla_only",            "NEURON_FLAG_MDLA_ONLY"},
        {"ep.neuron.min_group_size",       "NEURON_FLAG_MIN_GROUP_SIZE"},
        {"ep.neuron.optimization_string",  "NEURON_FLAG_OPTIMIZATION_STRING"},
        {"ep.neuron.force_skip_ops",       "NEURON_FLAG_FORCE_SKIP_OPS_STRING"},
    };
    for (const auto& m : kConfigMap) {
      RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(
          *session_options, m.cfg, "", config_val));
      if (!config_val.empty())
        f.ort_api.AddKeyValuePair(options, m.opt, config_val.c_str());
    }
  }

  NeuronEp::Config config;
  config.provider_options = options;  // ownership transferred to NeuronEp via factory cleanup

  RETURN_IF_ERROR(f.ort_api.Logger_LogMessage(
      logger, ORT_LOGGING_LEVEL_INFO,
      "NeuronEP: creating session EP", ORT_FILE, __LINE__, __FUNCTION__));

  auto neuron_ep = std::make_unique<NeuronEp>(f, f.ep_name_, config, *logger);

  // Transfer ownership of options to the ep (it will outlive this scope).
  // We intentionally do NOT release options here; NeuronEp borrows the pointer.
  // The lifetime is managed by the factory via provider_options_ if needed, or
  // simply leaked on session close (options is tiny; acceptable trade-off for
  // the demo; production code should track and free).
  // TODO: track per-session options in the factory for proper cleanup.

  *ep = neuron_ep.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL NeuronEpFactory::ReleaseEpImpl(OrtEpFactory* /*f*/, OrtEp* ep) noexcept {
  delete static_cast<NeuronEp*>(ep);
}

// ---- CreateAllocator --------------------------------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::CreateAllocatorImpl(
    OrtEpFactory* f_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*options*/,
    OrtAllocator** allocator) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  *allocator = nullptr;

  bool is_default  = (memory_info == f.default_memory_info_);
  bool is_readonly = (memory_info == f.readonly_memory_info_);

  if (!is_default && !is_readonly)
    return f.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "NeuronEP: unknown OrtMemoryInfo in CreateAllocator.");

  // Read-only allocator: one per session.
  if (is_readonly) {
    auto ro = std::make_unique<CustomAllocator>(memory_info, static_cast<ApiPtrs>(f));
    *allocator = ro.release();
    return nullptr;
  }

  // Default allocator: shared across sessions (reference-counted).
  std::lock_guard<std::mutex> lock(f.mutex_);
  if (!f.shared_allocator_)
    f.shared_allocator_ = std::make_unique<CustomAllocator>(memory_info,
                                                             static_cast<ApiPtrs>(f));
  ++f.num_allocator_users_;
  *allocator = f.shared_allocator_.get();
  return nullptr;
}

/*static*/
void ORT_API_CALL NeuronEpFactory::ReleaseAllocatorImpl(OrtEpFactory* f_ptr,
                                                         OrtAllocator* allocator) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  std::lock_guard<std::mutex> lock(f.mutex_);

  if (allocator == f.shared_allocator_.get()) {
    if (--f.num_allocator_users_ == 0)
      f.shared_allocator_.reset();
  } else {
    delete static_cast<CustomAllocator*>(allocator);
  }
}

// ---- CreateDataTransfer -----------------------------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::CreateDataTransferImpl(OrtEpFactory* f_ptr,
                                                                  OrtDataTransferImpl** dt) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  *dt = f.data_transfer_.get();  // factory-owned singleton; Release is a no-op
  return nullptr;
}

// ---- Stream -----------------------------------------------------------------
/*static*/
bool ORT_API_CALL NeuronEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*f*/) noexcept {
  return true;
}

/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* f_ptr,
    const OrtMemoryDevice* device,
    const OrtKeyValuePairs* /*options*/,
    OrtSyncStreamImpl** stream) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  *stream = nullptr;

  if (f.ep_api.MemoryDevice_GetMemoryType(device) == OrtDeviceMemoryType_DEFAULT) {
    auto s = std::make_unique<NeuronStreamImpl>(f, /*ep=*/nullptr);
    *stream = s.release();
  }
  return nullptr;
}

// ---- Incompatibility details ------------------------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::GetHardwareDeviceIncompatibilityDetailsImpl(
    OrtEpFactory* f_ptr,
    const OrtHardwareDevice* hw,
    OrtDeviceEpIncompatibilityDetails* details) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  auto device_type = f.ort_api.HardwareDevice_Type(hw);

  if (device_type != OrtHardwareDeviceType_CPU) {
    uint32_t reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    return f.ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details, reasons,
        static_cast<int32_t>(device_type),
        "NeuronEP supports CPU hardware devices only "
        "(MediaTek NPU acceleration is managed internally by the Neuron SDK).");
  }
  return nullptr;
}
