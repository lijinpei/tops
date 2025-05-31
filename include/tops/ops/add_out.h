#include <ATen/core/ATen_fwd.h>

namespace tops {
at::Tensor &cat_out(const at::Tensor &self, const at::Tensor &other,
                    const at::Scalar &value, at::Tensor &out);
}
