#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"
#include <tvm/ffi/extra/cuda/cubin_launcher.h>

#define uint8_t cake_dsv4_generated_uint8_t
#define uint16_t cake_dsv4_generated_uint16_t
#define uint32_t cake_dsv4_generated_uint32_t
#define uint64_t cake_dsv4_generated_uint64_t
#define int32_t cake_dsv4_generated_int32_t
#define int16_t cake_dsv4_generated_int16_t
#define CakeTensorMap cake_dsv4_generated_CakeTensorMap
#include "cake_dsv4_fp8_h64_source_exact.cu"
#undef CakeTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer::cake_dsv4 {

namespace {

static_assert(sizeof(CUtensorMap) == 128);
static_assert(sizeof(cake_dsv4_generated_CakeTensorMap) == sizeof(CUtensorMap));
static_assert(alignof(cake_dsv4_generated_CakeTensorMap) >= alignof(CUtensorMap));
static_assert(offsetof(cake_dsv4_generated_CakeTensorMap, opaque) == 0);
static_assert(sizeof(((cake_dsv4_generated_CakeTensorMap*)nullptr)->opaque) == sizeof(CUtensorMap));
static_assert(std::is_standard_layout<cake_dsv4_generated_CakeTensorMap>::value);
static_assert(std::is_trivially_copyable<cake_dsv4_generated_CakeTensorMap>::value);



using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError="
        << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:"
          << device_id << ": cudaError=" << static_cast<int>(error);
      restore_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
};

inline int64_t CakeDeviceMultiprocessorCount(int device_id) {
  constexpr int kMaxCachedCudaDevices = 64;
  TVM_FFI_CHECK(device_id >= 0 && device_id < kMaxCachedCudaDevices, RuntimeError)
      << "physical-SM-count cache does not cover cuda:" << device_id;
  static std::atomic<int> count_by_device[kMaxCachedCudaDevices]{};
  int cached = count_by_device[device_id].load(std::memory_order_acquire);
  if (cached > 0) return cached;

  int count = 0;
  cudaError_t error = cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, device_id);
  TVM_FFI_CHECK(error == cudaSuccess && count > 0, RuntimeError)
      << "querying multiProcessorCount failed for cuda:" << device_id
      << ": cudaError=" << static_cast<int>(error) << ", count=" << count;
  // Concurrent first launches may repeat the immutable device query, but all
  // publication is atomic and every later hot-path lookup is one acquire load.
  count_by_device[device_id].store(count, std::memory_order_release);
  return count;
}

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(
    const TensorView& t,
    const TensorView& reference,
    const char* name,
    const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id
      << " versus cuda:" << reference.device().device_id;
}

inline void CheckCurrentCudaDevice(
    const TensorView& reference,
    const char* reference_name) {
  int current_device = -1;
  cudaError_t error = cudaGetDevice(&current_device);
  TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed while validating " << reference_name
      << ": cudaError=" << static_cast<int>(error);
  TVM_FFI_CHECK(current_device == reference.device().device_id, ValueError)
      << "current CUDA device must match " << reference_name
      << ": current=cuda:" << current_device
      << ", tensor=cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

#include <dlfcn.h>

// CUDA 13.4 adds an oversized shared-memory mode (function/launch attribute
// SHARED_MEMORY_MODE = 3) that lets a kernel exceed the standard
// MaxSharedMemoryPerBlockOptin ceiling up to device attribute 150
// (OversizedSharedMemoryPerBlock), at the cost of an 8 KiB L1 carveout.
#if (TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDA_VERSION) && CUDA_VERSION >= 13040) || \
    (!TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDART_VERSION) && CUDART_VERSION >= 13040)
#define CAKE_HAS_OVERSIZED_SMEM 1
#else
#define CAKE_HAS_OVERSIZED_SMEM 0
#endif

// Per-device dynamic-SMEM opt-in. Returns true when launches on this device
// must carry the ALLOW_OVERSIZED shared-memory-mode launch attribute: the
// request exceeds the device's standard opt-in ceiling
// (MaxSharedMemoryPerBlockOptin, attribute 97) but fits the oversized ceiling
// (OversizedSharedMemoryPerBlock, attribute 150). Within the standard ceiling
// it sets MAX_DYNAMIC_SHARED_SIZE_BYTES for the device, mirroring
// CubinKernel::SetMaxDynamicSharedMemory. cache: 0 = unresolved,
// 1 = standard opt-in done, 2 = oversized mode required.
inline bool CakeConfigureDynamicSmem(tvm::ffi::CubinKernel& kernel, int device_id,
                                     int smem_bytes, signed char* cache, int cache_len) {
  namespace cuda_api = tvm::ffi::cuda_api;
  TVM_FFI_CHECK(device_id >= 0 && device_id < cache_len, RuntimeError)
      << "dynamic-SMEM opt-in cache does not cover cuda:" << device_id;
  if (cache[device_id] != 0) {
    return cache[device_id] == 2;
  }
  auto device = cuda_api::GetDeviceHandle(device_id);
  int optin_max = 0;
  cuda_api::ResultType err = cuda_api::GetDeviceAttribute(
      &optin_max,
      /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN /
         cudaDevAttrMaxSharedMemoryPerBlockOptin */
      cuda_api::DeviceAttrType(97), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
      << "querying MaxSharedMemoryPerBlockOptin failed for cuda:" << device_id;
  if (smem_bytes <= optin_max) {
    err = cuda_api::SetKernelMaxDynamicSharedMem(kernel.GetHandle(), smem_bytes, device);
    TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
        << "MAX_DYNAMIC_SHARED_SIZE_BYTES=" << smem_bytes << " rejected for cuda:" << device_id;
    cache[device_id] = 1;
    return false;
  }
#if CAKE_HAS_OVERSIZED_SMEM
  int oversized_max = 0;
  err = cuda_api::GetDeviceAttribute(
      &oversized_max,
      /* CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK /
         cudaDevAttrOversizedSharedMemoryPerBlock */
      cuda_api::DeviceAttrType(150), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess && oversized_max >= smem_bytes, RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id << " and the oversized ceiling is "
      << oversized_max << " B";
  // Also set the function-level shared-memory mode: profiler-instrumented
  // launches (ncu) honor only the function-level attribute for oversized
  // dynamic SMEM and fail with LaunchFailed on the launch attribute alone.
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  err = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
                             CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY,
                             kernel.GetHandle(), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
      << "setting SHARED_MEMORY_MODE=ALLOW_OVERSIZED failed for cuda:" << device_id
      << " (error " << static_cast<int>(err) << ")";
#else
  // The process cudart may predate 13.4 (e.g. torch's pip cudart 13.0) and
  // reject cudaFuncAttributeSharedMemoryMode with cudaErrorInvalidValue even
  // though the driver supports the mode, so call the driver entry point
  // directly: cudaKernel_t is interchangeable with CUkernel, the runtime
  // device ordinal is the CUdevice, and libcuda.so.1 is already loaded.
  {
    using CakeCuKernelSetAttributeFn = int (*)(int attrib, int val, void* kernel, int dev);
    static const auto cake_cu_kernel_set_attribute =
        reinterpret_cast<CakeCuKernelSetAttributeFn>(dlsym(RTLD_DEFAULT, "cuKernelSetAttribute"));
    TVM_FFI_CHECK(cake_cu_kernel_set_attribute != nullptr, RuntimeError)
        << "cuKernelSetAttribute not resolvable while enabling the oversized shared-memory mode";
    int drv_err = cake_cu_kernel_set_attribute(
        /* CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE */ 17,
        /* CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY */ 3,
        reinterpret_cast<void*>(kernel.GetHandle()), device_id);
    TVM_FFI_CHECK(drv_err == 0, RuntimeError)
        << "setting SHARED_MEMORY_MODE=ALLOW_OVERSIZED failed for cuda:" << device_id
        << " (driver error " << drv_err << ")";
  }
#endif
  cache[device_id] = 2;
  return true;
#else
  TVM_FFI_THROW(RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id
      << " and this CUDA toolkit predates the 13.4 oversized shared-memory mode";
#endif
}

// 4D TMA descriptor for buffer 'tmap_q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'tmap_q' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'tmap_q' trailing dims must be positive";
  TVM_FFI_CHECK(d2 % 64 == 0, ValueError)
      << "TMA source 'tmap_q' extent " << d2
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(d1), (uint64_t)(64), (uint64_t)((d2 / 64)), (uint64_t)(d3)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (128, 64, 1, 1) exceeds resolved global dims for 'tmap_q'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (64 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
      (uint64_t)((carrier_stride_1 * 8) / 8),
      (uint64_t)((carrier_stride_2 * 8) / 8),
  };
  uint32_t box_dim[4] = {128u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'tmap_q') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'tmap_swa_kv' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_swa_kv(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_swa_kv' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_swa_kv' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'tmap_swa_kv' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "tmap_swa_kv");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'tmap_swa_kv' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'tmap_swa_kv' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1) exceeds resolved global dims for 'tmap_swa_kv'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_swa_kv' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_swa_kv' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_swa_kv' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 1u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'tmap_swa_kv') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'tmap_compressed_kv' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_compressed_kv(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_compressed_kv' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_compressed_kv' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'tmap_compressed_kv' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "tmap_compressed_kv");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'tmap_compressed_kv' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'tmap_compressed_kv' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1) exceeds resolved global dims for 'tmap_compressed_kv'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_compressed_kv' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_compressed_kv' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_compressed_kv' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 1u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'tmap_compressed_kv') failed: CUresult=" << (int)r;
  return tm;
}

}  // namespace


void Run_fp8_h64_source_exact(TensorView arg_tmap_q, TensorView arg_tmap_swa_kv, TensorView arg_tmap_compressed_kv, TensorView arg_O, TensorView arg_cum_seq_lens_q, TensorView arg_sparse_indices, TensorView arg_sparse_topk_lens, TensorView arg_sinks, TensorView arg_bmm1_scale, TensorView arg_bmm2_scale, int64_t arg_num_heads, int64_t arg_sparse_topk, int64_t arg_has_sinks, int64_t arg_total_work_items, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError)
      << "cuda_stream must be non-negative";
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      static_cast<uintptr_t>(cuda_stream));
  DLDevice dev = arg_tmap_q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_tmap_q, "tmap_q");
  CheckDtype(arg_tmap_q, "tmap_q", 1, 8, 1);
  CheckContiguous(arg_tmap_q, "tmap_q");
  CheckCudaTensor(arg_tmap_swa_kv, "tmap_swa_kv");
  CheckDtype(arg_tmap_swa_kv, "tmap_swa_kv", 1, 8, 1);
  CheckCudaTensor(arg_tmap_compressed_kv, "tmap_compressed_kv");
  CheckDtype(arg_tmap_compressed_kv, "tmap_compressed_kv", 1, 8, 1);
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_cum_seq_lens_q, "cum_seq_lens_q");
  CheckDtype(arg_cum_seq_lens_q, "cum_seq_lens_q", 0, 32, 1);
  CheckContiguous(arg_cum_seq_lens_q, "cum_seq_lens_q");
  CheckCudaTensor(arg_sparse_indices, "sparse_indices");
  CheckDtype(arg_sparse_indices, "sparse_indices", 0, 32, 1);
  CheckContiguous(arg_sparse_indices, "sparse_indices");
  CheckCudaTensor(arg_sparse_topk_lens, "sparse_topk_lens");
  CheckDtype(arg_sparse_topk_lens, "sparse_topk_lens", 0, 32, 1);
  CheckContiguous(arg_sparse_topk_lens, "sparse_topk_lens");
  CheckCudaTensor(arg_sinks, "sinks");
  CheckDtype(arg_sinks, "sinks", 2, 32, 1);
  CheckContiguous(arg_sinks, "sinks");
  CheckCudaTensor(arg_bmm1_scale, "bmm1_scale");
  CheckDtype(arg_bmm1_scale, "bmm1_scale", 2, 32, 1);
  CheckContiguous(arg_bmm1_scale, "bmm1_scale");
  CheckCudaTensor(arg_bmm2_scale, "bmm2_scale");
  CheckDtype(arg_bmm2_scale, "bmm2_scale", 2, 32, 1);
  CheckContiguous(arg_bmm2_scale, "bmm2_scale");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sparse_topk >= -2147483648LL && arg_sparse_topk <= 2147483647LL, ValueError)
      << "scalar 'sparse_topk' value " << arg_sparse_topk
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_has_sinks >= -2147483648LL && arg_has_sinks <= 2147483647LL, ValueError)
      << "scalar 'has_sinks' value " << arg_has_sinks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_work_items >= -2147483648LL && arg_total_work_items <= 2147483647LL, ValueError)
      << "scalar 'total_work_items' value " << arg_total_work_items
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_tmap_swa_kv, arg_tmap_q, "tmap_swa_kv", "tmap_q");
  CheckSameCudaDevice(arg_tmap_compressed_kv, arg_tmap_q, "tmap_compressed_kv", "tmap_q");
  CheckSameCudaDevice(arg_O, arg_tmap_q, "O", "tmap_q");
  CheckSameCudaDevice(arg_cum_seq_lens_q, arg_tmap_q, "cum_seq_lens_q", "tmap_q");
  CheckSameCudaDevice(arg_sparse_indices, arg_tmap_q, "sparse_indices", "tmap_q");
  CheckSameCudaDevice(arg_sparse_topk_lens, arg_tmap_q, "sparse_topk_lens", "tmap_q");
  CheckSameCudaDevice(arg_sinks, arg_tmap_q, "sinks", "tmap_q");
  CheckSameCudaDevice(arg_bmm1_scale, arg_tmap_q, "bmm1_scale", "tmap_q");
  CheckSameCudaDevice(arg_bmm2_scale, arg_tmap_q, "bmm2_scale", "tmap_q");
  CheckCurrentCudaDevice(arg_tmap_q, "tmap_q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";


  CUtensorMap p_tmap_q = EncodeTma_tmap_q(arg_tmap_q);
  CUtensorMap p_tmap_swa_kv = EncodeTma_tmap_swa_kv(arg_tmap_swa_kv);
  CUtensorMap p_tmap_compressed_kv = EncodeTma_tmap_compressed_kv(arg_tmap_compressed_kv);
  void* p_O = arg_O.data_ptr();
  void* p_cum_seq_lens_q = arg_cum_seq_lens_q.data_ptr();
  void* p_sparse_indices = arg_sparse_indices.data_ptr();
  void* p_sparse_topk_lens = arg_sparse_topk_lens.data_ptr();
  void* p_sinks = arg_sinks.data_ptr();
  void* p_bmm1_scale = arg_bmm1_scale.data_ptr();
  void* p_bmm2_scale = arg_bmm2_scale.data_ptr();
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_sparse_topk = (int32_t)arg_sparse_topk;
  int32_t v_has_sinks = (int32_t)arg_has_sinks;
  int32_t v_total_work_items = (int32_t)arg_total_work_items;
  void* kargs[] = {&p_tmap_q, &p_tmap_swa_kv, &p_tmap_compressed_kv, &p_O, &p_cum_seq_lens_q, &p_sparse_indices, &p_sparse_topk_lens, &p_sinks, &p_bmm1_scale, &p_bmm2_scale, &v_num_heads, &v_sparse_topk, &v_has_sinks, &v_total_work_items};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);
  cudaError_t status = cudaSuccess;
  status = cudaFuncSetAttribute(kernel_cake_dsv4_fp8_h64_source_exact,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      203136);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_cake_dsv4_fp8_h64_source_exact) failed: "
      << cudaGetErrorString(status);
  status = cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_dsv4_fp8_h64_source_exact), grid, block, kargs,
      203136u, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_cake_dsv4_fp8_h64_source_exact launch failed: " << cudaGetErrorString(status);
}


}  // namespace flashinfer::cake_dsv4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_fp8_h64_source_exact, flashinfer::cake_dsv4::Run_fp8_h64_source_exact);
