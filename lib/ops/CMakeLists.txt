add_library(tops_kernels
  add_out.cpp
  bitonic_sort.cu
  cublas_mm.cpp)
target_link_libraries(tops_kernels PUBLIC nvidia::cutlass::cutlass CUDA::cuda_driver CUDA::cudart CUDA::cublasLt spdlog::spdlog ${TORCH_LIBRARIES} absl::hash absl::flat_hash_map fmt::fmt)
