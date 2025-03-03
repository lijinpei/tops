cmake_minimum_required(VERSION 3.30)

project(tops)

set(CXX_STANDARD 17)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit MODULE REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG
  HINTS "${LLVM_LIBRARY_DIR}/cmake/clang")
include(AddLLVM)

add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/fmt)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/inja)
add_subdirectory(third_party/msgpack-c)
add_subdirectory(third_party/nlohmann-json)

option(TOPS_ENABLE_PYTHOH_BINDING "enable python bindings for tops" "ON")

if (TOPS_ENABLE_PYTHOH_BINDING)
  find_package(Python REQUIRED)
  set (CMAKE_PREFIX_PATH "${Python_SITELIB}/torch/share/cmake/" "${CMAKE_PREFIX_PATH}")
  find_package(Torch CONFIG REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

find_package(OpenSSL MODULE REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(lib)
add_subdirectory(tools)
