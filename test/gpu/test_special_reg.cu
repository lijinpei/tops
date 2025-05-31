#include "tops/gpu/error_check.hpp"
#include "tops/gpu/special_reg.cuh"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "spdlog/spdlog.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

ABSL_FLAG(unsigned int, device, 0, "GPU device to run test on");

namespace {

CUdevice dev;

void init_gpu_device() {
  int dev_id = absl::GetFlag(FLAGS_device);
  CHECK_GPU_DRIVER_API_CALL(cuInit, 0);
  int dev_count;
  CHECK_GPU_DRIVER_API_CALL(cuDeviceGetCount, &dev_count);
  if (dev_id >= dev_count) {
    spdlog::error("request non exist device {} (only exists {} devices)",
                  dev_id, dev_count);
    throw std::runtime_error("invalid --device command line argument");
  }
  CHECK_GPU_DRIVER_API_CALL(cuDeviceGet, &dev, dev_id);
  CHECK_GPU_RUNTIME_API_CALL(cudaSetDevice, dev_id);
}

void finish_gpu_device() { CHECK_GPU_RUNTIME_API_CALL(cudaDeviceReset); }

__global__ void get_sm_id_kernel(int *result) {
  auto cta_id = tops::gpu::get_cta_x_id();
  int sm_id = tops::gpu::get_curr_phys_sm_id();
  result[cta_id] = sm_id;
}

} // namespace

TEST(SpecialReg, SM_ID) {
  int max_smem, sm_count;
  CHECK_GPU_DRIVER_API_CALL(
      cuDeviceGetAttribute, &max_smem,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
  CHECK_GPU_DRIVER_API_CALL(cuDeviceGetAttribute, &sm_count,
                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  CHECK_GPU_RUNTIME_API_CALL(cudaFuncSetAttribute, &get_sm_id_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_smem);
  thrust::device_vector<int> sm_ids(sm_count);
  get_sm_id_kernel<<<sm_count, 1, max_smem>>>(sm_ids.data().get());
  CHECK_GPU_DRIVER_API_CALL(cuStreamSynchronize, nullptr);
  thrust::host_vector<int> host_sm_ids = sm_ids;
  EXPECT_EQ(host_sm_ids.size(), sm_count);
  std::sort(host_sm_ids.begin(), host_sm_ids.end());
  for (int i = 0; i < sm_count; ++i) {
    EXPECT_EQ(i, host_sm_ids[i]);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  init_gpu_device();
  auto res = RUN_ALL_TESTS();
  finish_gpu_device();
  return res;
}
