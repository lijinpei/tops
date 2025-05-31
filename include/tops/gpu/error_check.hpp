#pragma once

#include "cuda.h"
#include "cuda_runtime_api.h"

namespace tops {
namespace gpu {

#define CHECK_GPU_DRIVER_API_CALL(func, ...)                                   \
  ::tops::gpu::assert_cuda_driver_api_result(func(__VA_ARGS__), #func,         \
                                             __FILE__, __LINE__)

#define CHECK_GPU_RUNTIME_API_CALL(func, ...)                                  \
  ::tops::gpu::assert_cuda_runtime_api_result(func(__VA_ARGS__), #func,        \
                                              __FILE__, __LINE__)

void assert_cuda_driver_api_result(CUresult result, const char *func,
                                   const char *file, int line);
void assert_cuda_runtime_api_result(cudaError_t result, const char *func,
                                    const char *file, int line);
} // namespace gpu
} // namespace tops
