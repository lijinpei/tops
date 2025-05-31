add_library(tops_gpu error_check.cpp)
add_subdirectory(ops)
target_link_libraries(tops_gpu PUBLIC tops_gpu_ops CUDA::cuda_driver CUDA::cudart spdlog::spdlog)
