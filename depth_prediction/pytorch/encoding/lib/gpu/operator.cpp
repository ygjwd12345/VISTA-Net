#include "operator.h"
#include <pybind11/pybind11.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregate_forward", &Aggregate_Forward_CUDA, "Aggregate forward (CUDA)");
  m.def("aggregate_backward", &Aggregate_Backward_CUDA, "Aggregate backward (CUDA)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CUDA, "ScaledL2 forward (CUDA)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CUDA, "ScaledL2 backward (CUDA)");
  m.def("batchnorm_forward", &BatchNorm_Forward_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("sumsquare_forward", &Sum_Square_Forward_CUDA, "SumSqu forward (CUDA)");
  m.def("sumsquare_backward", &Sum_Square_Backward_CUDA, "SumSqu backward (CUDA)");
}
