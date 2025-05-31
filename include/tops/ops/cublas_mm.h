#pragma once

#include <ATen/core/ATen_fwd.h>

namespace tops {
void cublas_mm_out(at::Tensor &d_mat, const at::Tensor &a_mat,
                   const at::Tensor &b_mat, const at::Tensor &c_mat,
                   const at::Tensor &alpha, const at::Tensor &beta,
                   bool trans_a = false, bool trans_b = false,
                   bool trans_c = false);
}
