add_library(tops_gpu error_check.cpp)
target_link_libraries(tops_gpu PUBLIC CUDA::cuda_driver CUDA::cudart spdlog::spdlog)
