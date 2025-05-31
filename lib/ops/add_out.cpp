#include "tops/ops/add_out.h"
#include "tops/utils/generated_kernel.h"

#include "ATen/Dispatch.h"
#include "ATen/Tensor.h"
#include "c10/core/ScalarType.h"

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

namespace tops {
namespace {

struct __attribute__((packed)) AddCacheKey {
  uint32_t BLOCK_SIZE;
  c10::ScalarType dtype;
  bool needs_mask;

  bool operator==(const AddCacheKey &other) const {
    return BLOCK_SIZE == other.BLOCK_SIZE && dtype == other.dtype &&
           needs_mask == other.needs_mask;
  }
};

template <typename H> H AbslHashValue(H h, const AddCacheKey &key) {
  return H::combine(
      std::move(h), key.BLOCK_SIZE,
      static_cast<std::underlying_type_t<c10::ScalarType>>(key.dtype),
      key.needs_mask);
}

struct AddKernel {
  LoadedKernelLib loaded;
  void ensure_kernel(const AddCacheKey &key) {
    if (__builtin_expect(loaded.launcher_func != nullptr, 1)) {
      return;
    }
    load_kernel(key);
  }
  void load_kernel(const AddCacheKey &key);
};

void AddKernel::load_kernel(const AddCacheKey &key) {}
} // namespace

void decide_kernel_config(size_t numel, c10::ScalarType dtype,
                          uint32_t &block_size, uint32_t &num_warps) {}

at::Tensor &add_out(const at::Tensor &self, const at::Tensor &other,
                    const at::Scalar &value, at::Tensor &out) {
  auto dtype = self.dtype().toScalarType();
  auto numel = self.numel();
  uint32_t num_warps;
  uint32_t block_size;
  decide_kernel_config(numel, dtype, block_size, num_warps);
  bool needs_mask = numel % block_size != 0;
  auto kernel_key = AddCacheKey{block_size, dtype, needs_mask};
  static absl::flat_hash_map<AddCacheKey, AddKernel> kernelCache;
  auto &cached_kernel = kernelCache[kernel_key];
  cached_kernel.ensure_kernel(kernel_key);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half,
                                         at::ScalarType::BFloat16, dtype,
                                         "add_out", [&]() {});
  return out;
}
} // namespace tops
