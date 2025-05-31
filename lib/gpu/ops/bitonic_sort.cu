#include "tops/gpu/error_check.hpp"
#include "tops/gpu/launch_grid.hpp"
#include "tops/gpu/ops/bitonic_sort.hpp"
#include "tops/gpu/ops/bitonic_sort_impl.cuh"
#include "tops/utils/int_numeric.hpp"

#include "ATen/ScalarType.h"
#include "ATen/Tensor.h"
#include "ATen/cuda/CUDAContext.h"

#include "cuda.h"

#include <cassert>
#include <cstdint>
#include <utility>

namespace tops {
namespace gpu {

namespace {
struct BitonicSortConfig {
  static constexpr uint8_t VecCount = 4;
  static constexpr uint8_t NumelPerThread = 8;
  static constexpr uint8_t LanesPerRow = 32;
  static constexpr uint8_t WarpsPerRow = 32;
  static constexpr uint8_t WarpsOnCol = 1;
};
LaunchGrid decide_launch_grid(size_t dim0, size_t dim1, uint32_t num_sm) {
  uint32_t num_warps;
  if (dim1 >= 32 * 128) {
    num_warps = 32;
  } else {
    num_warps = next_power_of_2(dim1 / 128);
  }
  uint32_t boost_factor = 32 / num_warps;
  uint32_t num_blocks = std::min<size_t>(boost_factor * num_sm, dim0);
  return {num_blocks, num_warps};
}
} // namespace

void bitonic_sort(const at::Tensor &self, std::optional<bool> stable,
                  int64_t dim, bool descending, const at::Tensor &values,
                  const at::Tensor &indices) {
  assert(self.dtype() == c10::kFloat);
  assert(values.dtype() == c10::kFloat);
  assert(indices.dtype() == c10::kLong);

  auto num_dim = self.dim();
  if (dim < 0) {
    dim += num_dim;
  }
  assert(dim + 1 == num_dim);
  assert(!stable.has_value() || !stable.value());

  auto self_cont = self.contiguous();
  auto numel = self.numel();
  auto row_size = self.size(dim);
  auto col_size = numel / row_size;

  auto *dev_prop = at::cuda::getCurrentDeviceProperties();
  auto num_sm = dev_prop->multiProcessorCount;
  auto launch_grid = decide_launch_grid(col_size, row_size, num_sm);
  kernels::bitonic_sort<BitonicSortConfig>
      <<<launch_grid.grid_shape, launch_grid.block_shape>>>(
          values.data_ptr<float>(), indices.data_ptr<int64_t>(),
          self.data_ptr<float>(), col_size, row_size);
  ;
  CHECK_GPU_RUNTIME_API_CALL(cudaGetLastError);
}

} // namespace gpu
} // namespace tops
