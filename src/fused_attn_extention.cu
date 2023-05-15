#include <torch/extension.h>

#include "fused_attn.cuh"

#define LAUNCH_WITH_ARGS(head_dim, chunk_size)                              \
    launch_attention_kernel<head_dim, chunk_size>(                          \
        batch_size, seq_len, num_features,                                  \
        (half_t*)queries.data_ptr<at::Half>(),                              \
        (half_t*)keys.data_ptr<at::Half>(),                                 \
        (half_t*)values.data_ptr<at::Half>(),                               \
        (half_t*)(mask.has_value() ? mask->data_ptr<at::Half>() : nullptr), \
        (half_t*)output.data_ptr<at::Half>()                                \
    )

torch::Tensor attention_forward(
    int head_dim, 
    int chunk_size, 
    torch::Tensor queries, 
    torch::Tensor keys, 
    torch::Tensor values, 
    torch::optional<torch::Tensor> mask
) {
    // Ensure inputs are of correct shape, dtype, and contiguous
    TORCH_CHECK(queries.dtype() == torch::kFloat16 && 
                queries.device().type() == torch::kCUDA && 
                queries.is_contiguous() &&
                queries.dim() == 3);
                
    TORCH_CHECK(keys.dtype() == torch::kFloat16 && 
                keys.device().type() == torch::kCUDA && 
                keys.is_contiguous() &&
                keys.dim() == 3);

    TORCH_CHECK(values.dtype() == torch::kFloat16 && 
                values.device().type() == torch::kCUDA && 
                values.is_contiguous() &&
                values.dim() == 3);

    TORCH_CHECK(!mask.has_value() ||
                (mask->dtype() == torch::kFloat16 && 
                 mask->device().type() == torch::kCUDA && 
                 mask->is_contiguous() &&
                 mask->dim() == 3));

    TORCH_CHECK(!(head_dim & (head_dim - 1)) && 16 <= head_dim && head_dim <= 128);
    TORCH_CHECK(!(chunk_size & (chunk_size - 1)) && 16 <= chunk_size && chunk_size <= 128 && chunk_size <= 2 * head_dim);
    TORCH_CHECK(!(chunk_size == 16 && head_dim == 128));

    // Retrieve input dimensions
    const uint32_t batch_size = queries.size(0);
    const uint32_t seq_len = queries.size(1);
    const uint32_t num_features = queries.size(2);

    // Check if other tensors have the same shape
    TORCH_CHECK(keys.size(0) == batch_size && 
                keys.size(1) == seq_len &&
                keys.size(2) == num_features);

    TORCH_CHECK(values.size(0) == batch_size && 
                values.size(1) == seq_len &&
                values.size(2) == num_features);

    TORCH_CHECK(seq_len % chunk_size == 0 &&
                num_features % head_dim == 0);
    
    // Allocate output tensor
    auto output = torch::empty(
        {batch_size, seq_len, num_features},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
    );
    
    switch (head_dim)
    {
    case 16:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(16, 16); break;
        case 32: LAUNCH_WITH_ARGS(16, 32); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 32:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(32, 16); break;
        case 32: LAUNCH_WITH_ARGS(32, 32); break;
        case 64: LAUNCH_WITH_ARGS(32, 64); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 64:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(64, 16); break;
        case 32: LAUNCH_WITH_ARGS(64, 32); break;
        case 64: LAUNCH_WITH_ARGS(64, 64); break;
        case 128: LAUNCH_WITH_ARGS(64, 128); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 128:
        switch (chunk_size)
        {
        case 32: LAUNCH_WITH_ARGS(128, 32); break;
        case 64: LAUNCH_WITH_ARGS(128, 64); break;
        case 128: LAUNCH_WITH_ARGS(128, 128); break;
        default: TORCH_CHECK(false);
        }
        break;
    default: TORCH_CHECK(false);
    }

    // Check if kernel invocation done well
    CHECK_LAST_CUDA_ERROR();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward",
          &attention_forward,
          py::arg("head_dim"),
          py::arg("chunk_size"),
          py::arg("queries"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("mask") = torch::optional<torch::Tensor>(),
          "Fused multihead attention");
}
