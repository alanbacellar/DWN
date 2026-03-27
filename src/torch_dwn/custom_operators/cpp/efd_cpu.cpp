#include <torch/extension.h>
#include <vector>

torch::Tensor efd_cpu_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts
) {
    // input: (batch_size, input_length)
    // mapping: (num_luts, n)
    // luts: (num_luts, 2^n)
    // output: (batch_size, num_luts)

    auto batch_size = input.size(0);
    auto num_luts = mapping.size(0);
    auto n = mapping.size(1);

    auto output = torch::zeros({batch_size, num_luts}, input.options());

    auto input_accessor = input.accessor<float, 2>();
    auto mapping_accessor = mapping.accessor<int, 2>();
    auto luts_accessor = luts.accessor<float, 2>();
    auto output_accessor = output.accessor<float, 2>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < num_luts; ++j) {
            uint32_t addr = 0;
            for (int64_t l = 0; l < n; ++l) {
                int64_t input_idx = mapping_accessor[j][l];
                if (input_accessor[b][input_idx] > 0) {
                    addr |= (1 << l);
                }
            }
            output_accessor[b][j] = luts_accessor[j][addr];
        }
    }

    return output;
}

std::vector<torch::Tensor> efd_cpu_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  const float alpha,
  const float beta,
  torch::Tensor output_grad
) {
    // Simplified backward pass - only compute LUT gradients
    // Full backward implementation would be more complex

    auto input_grad = torch::zeros_like(input);
    auto luts_grad = torch::zeros_like(luts);

    auto batch_size = input.size(0);
    auto num_luts = mapping.size(0);
    auto n = mapping.size(1);

    auto input_accessor = input.accessor<float, 2>();
    auto mapping_accessor = mapping.accessor<int, 2>();
    auto output_grad_accessor = output_grad.accessor<float, 2>();
    auto luts_grad_accessor = luts_grad.accessor<float, 2>();

    // Compute LUT gradients
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t j = 0; j < num_luts; ++j) {
            // Build address
            uint32_t addr = 0;
            for (int64_t l = 0; l < n; ++l) {
                int64_t input_idx = mapping_accessor[j][l];
                if (input_accessor[b][input_idx] > 0) {
                    addr |= (1 << l);
                }
            }
            luts_grad_accessor[j][addr] += output_grad_accessor[b][j];
        }
    }

    return {input_grad, luts_grad};
}

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor efd_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts) {
    CHECK_INPUT_CPU(input);
    CHECK_INPUT_CPU(mapping);
    CHECK_INPUT_CPU(luts);
    return efd_cpu_forward(input, mapping, luts);
};

std::vector<torch::Tensor> efd_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  const float alpha,
  const float beta,
  torch::Tensor output_grad) {
    CHECK_INPUT_CPU(input);
    CHECK_INPUT_CPU(mapping);
    CHECK_INPUT_CPU(luts);
    CHECK_INPUT_CPU(output_grad);
    return efd_cpu_backward(input, mapping, luts, alpha, beta, output_grad);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &efd_forward, "EFD CPU forward");
  m.def("backward", &efd_backward, "EFD CPU backward");
}