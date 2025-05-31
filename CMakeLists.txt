cmake_minimum_required(VERSION 3.22)

project(tops LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX)
find_package(Torch REQUIRED CONFIG HINTS "${TORCH_CMAKE_PREFIX}/../..")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CMAKE_FLAGS}")

add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)
add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/abseil-cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include")

add_subdirectory(lib)
add_subdirectory(test)
