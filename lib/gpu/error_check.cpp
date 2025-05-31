#include "tops/gpu/error_check.hpp"

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "spdlog/spdlog.h"

#include <stdexcept>

namespace tops {
namespace gpu {

void assert_cuda_driver_api_result(CUresult result, const char *func,
                                   const char *file, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *error_name = "<failed-to-get-error-name>";
  cuGetErrorName(result, &error_name);
  const char *error_desc = "\"failed to get error description\"";
  cuGetErrorString(result, &error_desc);
  spdlog::error("call cuda driver api {} failed with error code {} ({} : {}) "
                "at file {} +{}",
                func, (int)result, error_name, error_desc, file, line);
  throw std::runtime_error("cuda runtime api failed");
}

void assert_cuda_runtime_api_result(cudaError_t result, const char *func,
                                    const char *file, int line) {
  if (result == cudaSuccess) {
    return;
  }
  const char *error_name = cudaGetErrorName(result);
  const char *error_desc = cudaGetErrorString(result);
  spdlog::error("call cuda runtime api {} failed with error code {} ({} : {}) "
                "at file {} +{}",
                func, (int)result, error_name, error_desc, file, line);
  throw std::runtime_error("cuda runtime api failed");
}
} // namespace gpu
} // namespace tops
