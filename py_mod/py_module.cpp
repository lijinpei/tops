#include "pybind11/pybind11.h"

namespace py = pybind11;

void init_tops_cublasLt(py::module &&mod);

PYBIND11_MODULE(tops, m) {
  init_tops_cublasLt(
      m.def_submodule("cublasLt", "wrapper around cublasLt api"));
}
