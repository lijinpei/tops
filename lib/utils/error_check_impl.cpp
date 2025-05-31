#include "tops/utils/error_check.h"

#include "cublasLt.h"

#include "spdlog/spdlog.h"

#include <stdexcept>

namespace tops {

void assert_cuda_host_api_result_impl(CUresult result, const char *func,
                                      const char *file, int line) {
  const char *error_name = "<failed-to-get-cuda-driver-api-error-name>";
  cuGetErrorName(result, &error_name);
  const char *error_desc =
      "\"<failed-to-get-cuda-dirver-api-error-description>\"";
  cuGetErrorString(result, &error_desc);
  spdlog::error("cuda driver api {} failed with error code {} ({} : {}) "
                "at file {} +{}",
                func, (int)result, error_name, error_desc, file, line);
  throw std::runtime_error("cuda driver api failed");
}

void assert_cuda_host_api_result_impl(cudaError_t result, const char *func,
                                      const char *file, int line) {
  const char *error_name = cudaGetErrorName(result);
  const char *error_desc = cudaGetErrorString(result);
  spdlog::error("call cuda runtime api {} failed with error code {} ({} : {}) "
                "at file {} +{}",
                func, (int)result, error_name, error_desc, file, line);
  throw std::runtime_error("cuda runtime api failed");
}

void assert_cuda_host_api_result_impl(cublasStatus_t result, const char *func,
                                      const char *file, int line) {
  const char *error_name = cublasLtGetStatusName(result);
  const char *error_desc = cublasLtGetStatusString(result);
  spdlog::error("call cublas api {} failed with error code {} ({} : {}) "
                "at file {} +{}",
                func, (int)result, error_name, error_desc, file, line);
  throw std::runtime_error("cublas api failed");
}

} // namespace tops
