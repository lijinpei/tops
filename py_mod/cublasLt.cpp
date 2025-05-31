#include "pybind11/pybind11.h"

#include "torch/csrc/utils/pybind.h"

#include "tops/ops/cublas_mm.h"

#include "cublasLt.h"
#include "library_types.h"

namespace py = pybind11;

void init_tops_cublasLt(py::module &&mod) {
  mod.def("mm", tops::cublas_mm_out, py::arg("d_mat"), py::arg("a_mat"),
          py::arg("b_mat"), py::arg("c_mat"), py::arg("alpha"), py::arg("beta"),
          py::arg("trans_a") = false, py::arg("trans_b") = false,
          py::arg("trans_c") = false);
#define IMPL(x, y) value("" #y, CUBLAS_COMPUTE_##x)
  py::enum_<cublasComputeType_t>(mod, "compute_type")
      .IMPL(16F, f16)
      .IMPL(16F_PEDANTIC, f16_pedantic)
      .IMPL(32F, f32)
      .IMPL(32F_PEDANTIC, f32_pedantic)
      .IMPL(32F_FAST_16F, f32_fast_f16)
      .IMPL(32F_FAST_16BF, f32_fast_bf16)
      .IMPL(32F_FAST_TF32, f32_fast_tf32)
      .IMPL(32F_EMULATED_16BFX9, f32_emu_bf16x9)
      .IMPL(64F, f64)
      .IMPL(64F_PEDANTIC, f64_pedantic)
      .IMPL(32I, i32)
      .IMPL(32I_PEDANTIC, i32_pedantic);
#undef IMPL

#define IMPL(x, y) value("" #y, CUDA_##x)
  py::enum_<cudaDataType_t>(mod, "cuda_data_type")
      .IMPL(R_16F, f16)
      .IMPL(C_16F, complex_f16)
      .IMPL(R_16BF, bf16)
      .IMPL(C_16BF, complex_bf16)
      .IMPL(R_32F, f32)
      .IMPL(C_32F, complex_f32)
      .IMPL(R_64F, f64)
      .IMPL(C_64F, complex_f64)
      .IMPL(R_8I, s8)
      .IMPL(C_8I, complex_s8)
      .IMPL(R_8U, u8)
      .IMPL(C_8U, complex_u8)
      .IMPL(R_32I, s32)
      .IMPL(C_32I, complex_s32)
      .IMPL(R_8F_E4M3, f8_e4m3)
      .IMPL(R_8F_E5M2, f8_e5m2)
      .IMPL(R_4F_E2M1, f8_e2m1);
#undef IMPL

#define IMPL(x, y) value("" #y, CUBLASLT_POINTER_MODE_##x)
  py::enum_<cublasLtPointerMode_t>(mod, "pointer_mode")
      .IMPL(HOST, host)
      .IMPL(DEVICE, device)
      .IMPL(DEVICE_VECTOR, device_vector)
      .IMPL(ALPHA_DEVICE_VECTOR_BETA_ZERO, alpha_device_vector_beta_zero)
      .IMPL(ALPHA_DEVICE_VECTOR_BETA_HOST, alpha_device_vector_beta_host);
}
