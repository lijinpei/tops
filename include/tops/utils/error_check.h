#pragma once

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

namespace tops {

#define CHECK_CUDA(func, ...)                                                  \
  ::tops::assert_cuda_host_api_result(func(__VA_ARGS__), #func, __FILE__,      \
                                      __LINE__)

void assert_cuda_host_api_result_impl(CUresult result, const char *func,
                                      const char *file, int line);

void assert_cuda_host_api_result_impl(cudaError_t result, const char *func,
                                      const char *file, int line);

void assert_cuda_host_api_result_impl(cublasStatus_t result, const char *func,
                                      const char *file, int line);

inline bool assert_cuda_host_api_result(CUresult result, const char *func,
                                        const char *file, int line) {
  if (__builtin_expect(result == CUDA_SUCCESS, 1)) {
    return true;
  }
  assert_cuda_host_api_result_impl(result, func, file, line);
  return false;
}

inline bool assert_cuda_host_api_result(cudaError_t result, const char *func,
                                        const char *file, int line) {
  if (__builtin_expect(result == cudaSuccess, 1)) {
    return true;
  }
  assert_cuda_host_api_result_impl(result, func, file, line);
  return false;
}

inline bool assert_cuda_host_api_result(cublasStatus_t result, const char *func,
                                        const char *file, int line) {
  if (__builtin_expect(result == CUBLAS_STATUS_SUCCESS, 1)) {
    return true;
  }
  assert_cuda_host_api_result_impl(result, func, file, line);
  return false;
}

} // namespace tops
