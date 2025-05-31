add_library(tops_utils cublasLt_helpers.cpp error_check_impl.cpp exec_utils.cpp)
target_link_libraries(tops_utils PUBLIC CUDA::cuda_driver CUDA::cudart CUDA::cublasLt spdlog::spdlog)
