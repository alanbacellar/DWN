#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

__global__ void efd_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,    // (batch_size, input_lenght)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,    // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,     // (num_luts, 2^n)
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {       // (batch_size, num_luts)
    
    const int batch_size = output.size(0);
    const int num_luts = output.size(1);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_luts; j += blockDim.y * gridDim.y) {
                
            uint addr = input[i][mapping[j][0]] > 0;
            for(int l = 1; l < mapping.size(1); ++l)
                addr |= (uint)(input[i][mapping[j][l]] > 0) << l;

            output[i][j] = luts[j][addr];
    
        };
    };

};

torch::Tensor efd_cuda_forward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor) {
  
    auto batch_size = input_tensor.size(0);
    auto output_size = luts_tensor.size(0);

    auto output_tensor = torch::empty({batch_size, output_size}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, input_tensor.device().index()));

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    efd_cuda_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        output_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return output_tensor;
};

__global__ void efd_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,          // (batch_size, input_lenght)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,          // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,           // (num_luts, 2^n)
    const float alpha,
    const float beta,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_grad,    // (num_luts, n)
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_grad,           // (batch_size, input_lenght) 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts_grad) {          // (num_luts, 2^n)
          

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_grad.size(0); i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < output_grad.size(1); j += blockDim.y * gridDim.y) {

            // LUT grad
            uint addr = input[i][mapping[j][0]] > 0;
            for(int l = 1; l < mapping.size(1); ++l) {
                addr |= (uint)(input[i][mapping[j][l]] > 0) << l;
            };
            atomicAdd(&luts_grad[j][addr], output_grad[i][j]);

            // Input grad
            for(int l = 0; l < mapping.size(1); ++l) {
                float w = 0;
                uint addr_l_bit_off = addr & ~(1 << l);
                for(uint addr2 = 0; addr2 < luts.size(1); ++addr2) {
                    uint addr2_l_bit_off = addr2 & ~(1 << l);
                    auto dist = __popc(addr_l_bit_off ^ addr2_l_bit_off);
                    float fd = luts[j][addr2] * alpha *  __powf(beta, dist);
                    w += (addr2 >> l) & 1 ? fd : -fd;
                };
                atomicAdd(&input_grad[i][mapping[j][l]], w * output_grad[i][j]);
            };

        };
    };

};

std::vector<torch::Tensor> efd_cuda_backward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor,
    const float alpha,
    const float beta,
    torch::Tensor output_grad_tensor) {
  
    auto batch_size = output_grad_tensor.size(0);
    auto output_size = output_grad_tensor.size(1);

    auto input_grad_tensor = torch::zeros_like(input_tensor);
    auto luts_grad_tensor = torch::zeros_like(luts_tensor);

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    efd_cuda_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        alpha,
        beta,
        output_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        luts_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return {input_grad_tensor, luts_grad_tensor};
};