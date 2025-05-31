#pragma once

#include "tops/utils/error_check.h"

#include "ATen/core/ATen_fwd.h"
#include "c10/core/ScalarType.h"

#include "cublasLt.h"
#include "library_types.h"

#include <boost/core/noncopyable.hpp>
#include <cstdint>
#include <optional>

namespace tops {
namespace cublasLt {

std::optional<cudaDataType> toCudaDataType(c10::ScalarType dtype);

struct MatrixLayout : private boost::noncopyable {
  cublasLtMatrixLayout_t handle = nullptr;
  MatrixLayout(cublasLtMatrixLayout_t handle) : handle(handle) {}
  ~MatrixLayout();
};

MatrixLayout getCuBlasLtMatrixLayout(const at::Tensor &mat, int64_t batch);

class MatmulDesc : boost::noncopyable {
  cublasLtMatmulDesc_t handle = nullptr;
  c10::ScalarType compute_type;
  c10::ScalarType scale_type;

public:
  MatmulDesc(cublasLtMatmulDesc_t handle) : handle(handle) {}
  cublasLtMatmulDesc_t getHandle() const { return handle; }
  c10::ScalarType getComputeType() const { return compute_type; }
  c10::ScalarType getScaleType() const { return scale_type; }
  ~MatmulDesc();
};

std::optional<MatmulDesc> createMatmulDesc(
    at::Tensor &d_mat, const at::Tensor &a_mat, const at::Tensor &b_mat,
    const at::Tensor &c_mat, const at::Tensor &alpha, const at::Tensor &beta,
    bool trans_a = false, bool trans_b = false, bool trans_c = false);

class MatmulPreference : boost::noncopyable {
  cublasLtMatmulPreference_t handle = nullptr;

public:
  MatmulPreference(cublasLtMatmulPreference_t handle) : handle(handle) {}
  MatmulPreference() {
    if (!CHECK_CUDA(cublasLtMatmulPreferenceCreate, &handle)) {
      handle = nullptr;
      return;
    }
    CHECK_CUDA(cublasLtMatmulPreferenceInit, handle);
  }
  ~MatmulPreference() {
    if (handle) {
      CHECK_CUDA(cublasLtMatmulPreferenceDestroy, handle);
    }
  }
  cublasLtMatmulPreference_t getHandle() const { return handle; }

#define CUBLAS_MATMUL_MEMBER_FUNC_IMPL(DTYPE, VAR_NAME, ENUM)                  \
  DTYPE get##VAR_NAME() const {                                                \
    DTYPE VAR_NAME;                                                            \
    size_t sizeWritten;                                                        \
    (void)sizeWritten;                                                         \
    CHECK_CUDA(cublasLtMatmulPreferenceGetAttribute, handle, ENUM, &VAR_NAME,  \
               sizeof(VAR_NAME), &sizeWritten);                                \
    assert(sizeWritten == sizeof(VAR_NAME));                                   \
    return VAR_NAME;                                                           \
  }                                                                            \
                                                                               \
  void set##VAR_NAME(DTYPE VAR_NAME) {                                         \
    CHECK_CUDA(cublasLtMatmulPreferenceSetAttribute, handle, ENUM, &VAR_NAME,  \
               sizeof(VAR_NAME));                                              \
  }

  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint32_t, SearchMode,
                                 CUBLASLT_MATMUL_PREF_SEARCH_MODE)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint64_t, MaxWorkspaceBytes,
                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint32_t, ReductionSchemeMask,
                                 CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint32_t, MinAlignmentABytes,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint32_t, MinAlignmentBBytes,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint32_t, MinAlignmentCBytes,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint32_t, MinAlignmentDBytes,
                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(float, MaxWaveCount,
                                 CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT)
  CUBLAS_MATMUL_MEMBER_FUNC_IMPL(uint64_t, ImplMask,
                                 CUBLASLT_MATMUL_PREF_IMPL_MASK)

#undef CUBLAS_MATMUL_MEMBER_FUNC_IMPL
};

} // namespace cublasLt
} // namespace tops
