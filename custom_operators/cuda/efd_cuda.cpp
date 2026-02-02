#include <torch/extension.h>

#include <vector>

torch::Tensor efd_cuda_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts
);

std::vector<torch::Tensor> efd_cuda_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  const float alpha,
  const float beta,
  torch::Tensor output_grad
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor efd_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts) {
    CHECK_INPUT(input);
    CHECK_INPUT(mapping);
    CHECK_INPUT(luts);
    return efd_cuda_forward(input, mapping, luts);
};

std::vector<torch::Tensor> efd_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  const float alpha,
  const float beta,
  torch::Tensor output_grad) {
    CHECK_INPUT(input);
    CHECK_INPUT(mapping);
    CHECK_INPUT(luts);
    CHECK_INPUT(output_grad);
    return efd_cuda_backward(input, mapping, luts, alpha, beta, output_grad);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &efd_forward, "EFD CUDA forward");
  m.def("backward", &efd_backward, "EFD CUDA backward");
}