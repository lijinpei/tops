#include "tops/utils/cublasLt_helpers.h"
#include "tops/utils/error_check.h"

#include "ATen/core/Tensor.h"

namespace tops {
namespace cublasLt {

std::optional<cudaDataType> toCudaDataType(c10::ScalarType dtype) {
  switch (dtype) {
  default:
    break;
  case c10::ScalarType::Half:
    return CUDA_R_16F;
  case c10::ScalarType::BFloat16:
    return CUDA_R_16BF;
  case c10::ScalarType::Float:
    return CUDA_R_32F;
  case c10::ScalarType::Double:
    return CUDA_R_64F;
  case c10::ScalarType::Char:
    return CUDA_R_8I;
  case c10::ScalarType::Byte:
    return CUDA_R_8U;
  case c10::ScalarType::Int:
    return CUDA_R_32I;
  case c10::ScalarType::Float8_e5m2:
    return CUDA_R_8F_E5M2;
  }
  return {};
}

MatrixLayout::~MatrixLayout() {
  if (handle) {
    CHECK_CUDA(cublasLtMatrixLayoutDestroy, handle);
  }
}

MatrixLayout getCuBlasLtMatrixLayout(const at::Tensor &mat, int64_t batch) {
  int64_t dim = mat.dim();
  auto dim0 = (uint64_t)mat.size(dim - 2);
  auto dim1 = (uint64_t)mat.size(dim - 1);
  int64_t stride0 = mat.stride(dim - 2);
  int64_t stride1 = mat.stride(dim - 1);
  int64_t ld;
  assert(stride0 == 1 || stride1 == 1);
  int32_t order;
  if (stride1 == 1) {
    ld = stride0;
    order = CUBLASLT_ORDER_ROW;
  } else {
    ld = stride1;
    order = CUBLASLT_ORDER_COL;
  }
  cublasLtMatrixLayout_t mat_layout = nullptr;
  auto maybe_type = toCudaDataType(mat.scalar_type());
  assert(maybe_type.has_value());
  CHECK_CUDA(cublasLtMatrixLayoutCreate, &mat_layout, *maybe_type, dim0, dim1,
             ld);
  CHECK_CUDA(cublasLtMatrixLayoutSetAttribute, mat_layout,
             CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (batch != 1) {
    auto batch_i32 = (int32_t)batch;
    CHECK_CUDA(cublasLtMatrixLayoutSetAttribute, mat_layout,
               CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_i32,
               sizeof(batch_i32));
    int64_t batch_stride = mat.stride(0);
    CHECK_CUDA(cublasLtMatrixLayoutSetAttribute, mat_layout,
               CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
               sizeof(batch_stride));
    int32_t batch_mode = CUBLASLT_BATCH_MODE_STRIDED;
    CHECK_CUDA(cublasLtMatrixLayoutSetAttribute, mat_layout,
               CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode,
               sizeof(batch_mode));
  }
  return {mat_layout};
}

MatmulDesc::~MatmulDesc() {
  if (handle) {
    CHECK_CUDA(cublasLtMatmulDescDestroy, handle);
  }
}

std::optional<MatmulDesc>
createMatmulDesc(at::Tensor &d_mat, const at::Tensor &a_mat,
                 const at::Tensor &b_mat, const at::Tensor &c_mat,
                 const at::Tensor &alpha, const at::Tensor &beta, bool trans_a,
                 bool trans_b, bool trans_c) {
  auto alpha_dtype = alpha.scalar_type();
  assert(alpha_dtype == beta.scalar_type());
  auto scale_type = toCudaDataType(alpha_dtype);
  if (!scale_type.has_value()) {
    return {};
  }
  auto a_dtype = a_mat.scalar_type();
  assert(a_dtype == b_mat.scalar_type());
  cublasComputeType_t compute_type;
  switch (a_dtype) {
  default:
    return {};
  case c10::ScalarType::Half:
    compute_type = CUBLAS_COMPUTE_16F;
  case c10::ScalarType::BFloat16:
    compute_type = CUBLAS_COMPUTE_32F;
  case c10::ScalarType::Float:
    compute_type = CUBLAS_COMPUTE_32F;
  case c10::ScalarType::Double:
    compute_type = CUBLAS_COMPUTE_64F;
  case c10::ScalarType::Char:
    compute_type = CUBLAS_COMPUTE_32I;
  case c10::ScalarType::Byte:
    compute_type = CUBLAS_COMPUTE_32I;
  case c10::ScalarType::Int:
    compute_type = CUBLAS_COMPUTE_32I;
  case c10::ScalarType::Float8_e5m2:
    compute_type = CUBLAS_COMPUTE_32F;
  };
  cublasLtMatmulDesc_t handle = nullptr;
  CHECK_CUDA(cublasLtMatmulDescCreate, &handle, compute_type, *scale_type);
  bool alpha_is_cuda = alpha.is_cuda();
  assert(alpha_is_cuda == beta.is_cuda());
  int32_t pointer_mode =
      alpha_is_cuda ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
  CHECK_CUDA(cublasLtMatmulDescSetAttribute, handle,
             CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode,
             sizeof(pointer_mode));
  int32_t trans = CUBLAS_OP_T;
  if (trans_a) {
    CHECK_CUDA(cublasLtMatmulDescSetAttribute, handle,
               CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
  }
  if (trans_b) {
    CHECK_CUDA(cublasLtMatmulDescSetAttribute, handle,
               CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans));
  }
  if (trans_c) {
    CHECK_CUDA(cublasLtMatmulDescSetAttribute, handle,
               CUBLASLT_MATMUL_DESC_TRANSC, &trans, sizeof(trans));
  }
  return {handle};
}

} // namespace cublasLt
} // namespace tops
