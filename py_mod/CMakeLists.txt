pybind11_add_module(tops py_module.cpp cublasLt.cpp)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(tops PUBLIC tops_kernels tops_utils pybind11::pybind11 torch ${TORCH_PYTHON_LIBRARY})
