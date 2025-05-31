#pragma once

namespace tops {

namespace kernels {
template <typename Config, typename ElemTy, typename IndiceTy>
__global__ void bitonic_sort(ElemTy *outp_values, IndiceTy *outp_indices,
                             const ElemTy *inp_values, uint32_t dim0,
                             uint32_t dim1) {}
} // namespace kernels
} // namespace tops
