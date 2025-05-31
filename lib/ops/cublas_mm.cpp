#include "tops/ops/cublas_mm.h"
#include "tops/utils/cublasLt_helpers.h"
#include "tops/utils/error_check.h"

#include "ATen/core/Tensor.h"
#include "c10/cuda/CUDAGuard.h"

#include "cublasLt.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace tops {
namespace {

struct CUBLASHandle {
  c10::DeviceIndex dev_id;
  cublasLtHandle_t handle;
  CUBLASHandle(c10::DeviceIndex dev_id) : dev_id(dev_id) {
    CHECK_CUDA(cublasLtCreate, &handle);
  }
  ~CUBLASHandle() { CHECK_CUDA(cublasLtDestroy, handle); }
};

std::unordered_map<c10::DeviceIndex, CUBLASHandle> cublas_handles;

CUBLASHandle &ensure_cublas_handle_from_device(c10::DeviceIndex dev_id) {
  auto iter = cublas_handles.find(dev_id);
  if (iter == cublas_handles.end()) {
    iter = cublas_handles.try_emplace(iter, dev_id, dev_id);
  }
  return iter->second;
}

} // namespace

void cublas_mm_out(at::Tensor &d_mat, const at::Tensor &a_mat,
                   const at::Tensor &b_mat, const at::Tensor &c_mat,
                   const at::Tensor &alpha, const at::Tensor &beta,
                   bool trans_a, bool trans_b, bool trans_c) {
  assert(d_mat.is_cuda());
  assert(a_mat.is_cuda());
  assert(b_mat.is_cuda());
  assert(c_mat.is_cuda());
  auto dev_id = a_mat.get_device();
  c10::cuda::CUDAGuard device_guard(dev_id);
  auto &cublas_handle = ensure_cublas_handle_from_device(dev_id);
  bool scale_factor_host = alpha.is_cuda();
  assert(beta.is_cuda() == scale_factor_host);
  int64_t matmul_dim;
  int64_t a_dim = a_mat.dim();
  assert(a_dim == 3 || a_dim == 2);
  int64_t a_batch;
  if (a_dim == 3) {
    a_batch = a_mat.size(0);
    matmul_dim = 3;
  } else {
    a_batch = 1;
    matmul_dim = 2;
  }
  int64_t b_dim = b_mat.dim();
  assert(b_dim == 3 || a_dim == 2);
  int64_t b_batch;
  if (b_dim == 3) {
    b_batch = b_mat.size(0);
    matmul_dim = 3;
  } else {
    b_batch = 1;
  }
  int64_t c_dim = c_mat.dim();
  (void)c_dim;
  assert(c_dim == 2 || c_dim == 3);
  assert(c_dim <= matmul_dim);
  int64_t d_dim = d_mat.dim();
  (void)d_dim;
  assert(d_dim == matmul_dim);
  int64_t batch = std::max(a_batch, b_batch);
  assert(matmul_dim == 2 || d_mat.size(0) == batch);
  auto a_layout = cublasLt::getCuBlasLtMatrixLayout(a_mat, batch);
  auto b_layout = cublasLt::getCuBlasLtMatrixLayout(b_mat, batch);
  auto c_layout = cublasLt::getCuBlasLtMatrixLayout(c_mat, batch);
  auto d_layout = cublasLt::getCuBlasLtMatrixLayout(d_mat, batch);
  auto matmul_desc = cublasLt::createMatmulDesc(
      d_mat, a_mat, b_mat, c_mat, alpha, beta, trans_a, trans_b, trans_c);
  cublasLt::MatmulPreference matmulPref;
  int num_matmul_heurs;
  cublasLtMatmulHeuristicResult_t matmul_heur;
  CHECK_CUDA(cublasLtMatmulAlgoGetHeuristic, cublas_handle.handle,
             matmul_desc->getHandle(), a_layout.handle, b_layout.handle,
             c_layout.handle, d_layout.handle, matmulPref.getHandle(), 1,
             &matmul_heur, &num_matmul_heurs);
  CHECK_CUDA(cublasLtMatmul, cublas_handle.handle, matmul_desc->getHandle(),
             alpha.data_ptr(), a_mat.data_ptr(), a_layout.handle,
             b_mat.data_ptr(), b_layout.handle, beta.data_ptr(),
             c_mat.data_ptr(), c_layout.handle, d_mat.data_ptr(),
             d_layout.handle, &matmul_heur.algo, nullptr, 0, nullptr);
}

} // namespace tops
