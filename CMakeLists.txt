cmake_minimum_required(VERSION 3.22)

set(TORCH_CUDA_ARCH_LIST 8.0 8.9 9.0a)

project(tops LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Xcompiler -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

find_package (Python COMPONENTS Development)

include(FindCUDAToolkit)
execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX)
set(USE_SYSTEM_NVTX ON CACHE BOOL "")
find_package(Torch REQUIRED CONFIG HINTS "${TORCH_CMAKE_PREFIX}/../..")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CMAKE_FLAGS}")
include_directories("${TORCH_INCLUDE_DIRS}")

add_subdirectory(third_party)
include(GoogleTest)
enable_testing()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(lib)
add_subdirectory(py_mod)
add_subdirectory(test)
