add_library(tops_gpu_ops bitonic_sort.cu)
target_link_libraries(tops_gpu_ops PUBLIC CUDA::cuda_driver CUDA::cudart spdlog::spdlog ${TORCH_LIBRARIES})
