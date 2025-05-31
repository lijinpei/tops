#include <ATen/core/ATen_fwd.h>

#include <cstdint>
#include <utility>

namespace tops {
namespace gpu {
void bitonic_sort(const at::Tensor &self, std::optional<bool> stable,
                  int64_t dim, bool descending, const at::Tensor &values,
                  const at::Tensor &indices);
}
} // namespace tops
