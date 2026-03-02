// neuron_ep_factory.cc

#include "neuron_ep_factory.h"

#include <cassert>
#include "neuron_ep.h"
#include "neuron_ep_stream_support.h"

// ===========================================================================
// Constructor
// ===========================================================================
NeuronEpFactory::NeuronEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_{default_logger},
      ep_name_{ep_name},
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr} {
  // Declare the ORT API version we were compiled against.
  ort_version_supported = ORT_API_VERSION;

  // Populate the C vtable.
  GetName               = GetNameImpl;
  GetVendor             = GetVendorImpl;
  GetVendorId           = GetVendorIdImpl;
  GetVersion            = GetVersionImpl;
  GetSupportedDevices   = GetSupportedDevicesImpl;
  CreateEp              = CreateEpImpl;
  ReleaseEp             = ReleaseEpImpl;
  CreateAllocator       = CreateAllocatorImpl;
  ReleaseAllocator      = ReleaseAllocatorImpl;
  CreateDataTransfer    = CreateDataTransferImpl;
  IsStreamAware         = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

  // Describe the fake "device" this EP exposes.
  // We pretend to be GPU to exercise the full API (same as example_plugin_ep).
  // A real NPU/CPU EP would use OrtMemoryInfoDeviceType_CPU.
  default_memory_info_ = Ort::MemoryInfo{
      "NeuronDummy GPU",
      OrtMemoryInfoDeviceType_GPU,
      /*vendor*/    0xABCD,
      /*device_id*/ 0,
      OrtDeviceMemoryType_DEFAULT,
      /*alignment*/ 0,
      OrtAllocatorType::OrtDeviceAllocator};

  readonly_memory_info_ = Ort::MemoryInfo{
      "NeuronDummy GPU readonly",
      OrtMemoryInfoDeviceType_GPU,
      /*vendor*/    0xABCD,
      /*device_id*/ 0,
      OrtDeviceMemoryType_DEFAULT,
      /*alignment*/ 0,
      OrtAllocatorType::OrtReadOnlyAllocator};

  // Create data-transfer object (shared across all sessions).
  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_ = std::make_unique<NeuronDataTransfer>(apis, device);
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
// ORT passes all available hardware devices; we claim the first CPU device.
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
    // Claim CPU hardware devices.
    if (f.ort_api.HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) continue;

    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options  = nullptr;
    f.ort_api.CreateKeyValuePairs(&ep_metadata);
    f.ort_api.CreateKeyValuePairs(&ep_options);

    // Metadata visible to users querying the registered EP.
    f.ort_api.AddKeyValuePair(ep_metadata, "version",       f.ep_version_.c_str());
    f.ort_api.AddKeyValuePair(ep_metadata, "ep_name",       "NeuronDummyPluginEP");
    f.ort_api.AddKeyValuePair(ep_options,  "neuron_option", "example_value");

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
// Called once per session.  Creates a NeuronEp instance.
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::CreateEpImpl(
    OrtEpFactory* f_ptr,
    const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* /*ep_metadata*/,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  *ep = nullptr;

  if (num_devices != 1)
    return f.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "NeuronDummyEP: expected exactly one device.");

  RETURN_IF_ERROR(f.ort_api.Logger_LogMessage(
      logger, ORT_LOGGING_LEVEL_INFO,
      "NeuronDummyEP: creating session EP", ORT_FILE, __LINE__, __FUNCTION__));

  // Read session config.  Add your own config keys here.
  NeuronEp::Config config{};
  // Example: std::string val; GetSessionConfigEntryOrDefault(*session_options, "neuron.key", "0", val);
  (void)session_options;

  auto neuron_ep = std::make_unique<NeuronEp>(f, f.ep_name_, config, *logger);
  *ep = neuron_ep.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL NeuronEpFactory::ReleaseEpImpl(OrtEpFactory* /*f*/, OrtEp* ep) noexcept {
  delete static_cast<NeuronEp*>(ep);
}

// ---- CreateAllocator --------------------------------------------------------
// Returns a shared CustomAllocator for the default memory type.
// Returns a fresh CustomAllocator for the read-only (initializer) memory type.
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
                                  "NeuronDummyEP: unknown OrtMemoryInfo in CreateAllocator.");

  // Read-only allocator: one per session, not shared.
  if (is_readonly) {
    auto ro = std::make_unique<CustomAllocator>(memory_info, static_cast<ApiPtrs>(f));
    *allocator = ro.release();
    return nullptr;
  }

  // Default allocator: shared across sessions (reference-counted).
  std::lock_guard<std::mutex> lock(f.mutex_);
  if (!f.shared_allocator_) {
    f.shared_allocator_ = std::make_unique<CustomAllocator>(memory_info,
                                                             static_cast<ApiPtrs>(f));
  }
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
    // read-only allocator – delete directly.
    delete static_cast<CustomAllocator*>(allocator);
  }
}

// ---- CreateDataTransfer -----------------------------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEpFactory::CreateDataTransferImpl(OrtEpFactory* f_ptr,
                                                                  OrtDataTransferImpl** dt) noexcept {
  auto& f = *static_cast<NeuronEpFactory*>(f_ptr);
  *dt = f.data_transfer_.get();  // shared, factory-owned; ReleaseImpl is a no-op
  return nullptr;
}

// ---- Stream -----------------------------------------------------------------
/*static*/
bool ORT_API_CALL NeuronEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*f*/) noexcept {
  return true;  // we implement the stream interface (as no-ops for this CPU EP)
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
  auto  device_type = f.ort_api.HardwareDevice_Type(hw);

  if (device_type != OrtHardwareDeviceType_CPU) {
    uint32_t reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    return f.ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details, reasons,
        static_cast<int32_t>(device_type),
        "NeuronDummyEP only supports CPU devices");
  }
  return nullptr;
}
