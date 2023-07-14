#include <torch/extension.h>

#include "fused_attn.cuh"

#define LAUNCH_WITH_ARGS(head_dim, chunk_size)                              \
    launch_attention_kernel<head_dim, chunk_size>(                          \
        batch_size, seq_len, num_features,                                  \
        (half_t*)queries.data_ptr<at::Half>(),                              \
        (half_t*)keys.data_ptr<at::Half>(),                                 \
        (half_t*)values.data_ptr<at::Half>(),                               \
        (half_t*)(mask.has_value() ? mask->data_ptr<at::Half>() : nullptr), \
        (half_t*)output.data_ptr<at::Half>(),                               \
        (half_t*)(scores_max.has_value() ? scores_max->data_ptr<at::Half>() : nullptr), \
        (half_t*)(scores_sum.has_value() ? scores_sum->data_ptr<at::Half>() : nullptr)  \
    )

std::vector<torch::Tensor> attention_forward(
    int head_dim, 
    int chunk_size, 
    torch::Tensor queries, 
    torch::Tensor keys,
    torch::Tensor values, 
    torch::optional<torch::Tensor> mask,
    bool inference
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

    const uint32_t num_heads = num_features / head_dim;

    // Check if other tensors have the same shape
    TORCH_CHECK(keys.size(0) == batch_size && 
                keys.size(1) == seq_len &&
                keys.size(2) == num_features);

    TORCH_CHECK(values.size(0) == batch_size && 
                values.size(1) == seq_len &&
                values.size(2) == num_features);

    TORCH_CHECK(seq_len % chunk_size == 0 &&
                num_features % head_dim == 0);

    auto opts = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(torch::kCUDA);

    torch::optional<torch::Tensor> scores_max;
    torch::optional<torch::Tensor> scores_sum;

    if (!inference) {
        scores_max = torch::empty({batch_size, num_heads, seq_len}, opts);
        scores_sum = torch::empty({batch_size, num_heads, seq_len}, opts);
    }
    
    // Allocate output tensor
    auto output = torch::empty({batch_size, seq_len, num_features}, opts);
    
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

    if (inference) {
        return { output };
    }

    return { output, scores_max.value(), scores_sum.value() };
}

std::vector<torch::Tensor> attention_backward(
    int head_dim, 
    int chunk_size, 
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor output_grad,
    torch::Tensor scores_max,
    torch::Tensor scores_sum
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

    TORCH_CHECK(output_grad.dtype() == torch::kFloat16 && 
                output_grad.device().type() == torch::kCUDA && 
                output_grad.is_contiguous() &&
                output_grad.dim() == 3);

    TORCH_CHECK(scores_max.dtype() == torch::kFloat16 && 
                scores_max.device().type() == torch::kCUDA && 
                scores_max.is_contiguous() &&
                scores_max.dim() == 3);

    TORCH_CHECK(scores_sum.dtype() == torch::kFloat16 && 
                scores_sum.device().type() == torch::kCUDA && 
                scores_sum.is_contiguous() &&
                scores_sum.dim() == 3);

    TORCH_CHECK(!(head_dim & (head_dim - 1)) && 16 <= head_dim && head_dim <= 128);
    TORCH_CHECK(!(chunk_size & (chunk_size - 1)) && 16 <= chunk_size && chunk_size <= 128 && chunk_size <= 2 * head_dim);
    TORCH_CHECK(!(chunk_size == 16 && head_dim == 128));

    // Retrieve input dimensions
    const uint32_t batch_size = queries.size(0);
    const uint32_t seq_len = queries.size(1);
    const uint32_t num_features = queries.size(2);

    const uint32_t num_heads = num_features / head_dim;

    // Check if other tensors have the same shape
    TORCH_CHECK(keys.size(0) == batch_size && 
                keys.size(1) == seq_len &&
                keys.size(2) == num_features);

    TORCH_CHECK(values.size(0) == batch_size && 
                values.size(1) == seq_len &&
                values.size(2) == num_features);

    TORCH_CHECK(output_grad.size(0) == batch_size && 
                output_grad.size(1) == seq_len &&
                output_grad.size(2) == num_features);

    TORCH_CHECK(scores_max.size(0) == batch_size && 
                scores_max.size(1) == num_heads &&
                scores_max.size(2) == seq_len);

    TORCH_CHECK(scores_sum.size(0) == batch_size && 
                scores_sum.size(1) == num_heads &&
                scores_sum.size(2) == seq_len);

    TORCH_CHECK(seq_len % chunk_size == 0 &&
                num_features % head_dim == 0);

    auto opts = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(torch::kCUDA);

    torch::Tensor scores_grad_sum = torch::empty({batch_size, num_heads, seq_len}, opts);
    torch::Tensor queries_grad = torch::empty({batch_size, seq_len, num_features}, opts);
    torch::Tensor keys_grad = torch::empty({batch_size, seq_len, num_features}, opts);
    torch::Tensor values_grad = torch::empty({batch_size, seq_len, num_features}, opts);
    
    launch_attention_backward_kernel_a<128, 64>(
        batch_size, seq_len, num_features,
        (half_t*)queries.data_ptr<at::Half>(),
        (half_t*)keys.data_ptr<at::Half>(),
        (half_t*)values.data_ptr<at::Half>(),
        (half_t*)output_grad.data_ptr<at::Half>(),
        (half_t*)scores_max.data_ptr<at::Half>(),
        (half_t*)scores_sum.data_ptr<at::Half>(),
        (half_t*)scores_grad_sum.data_ptr<at::Half>(),
        (half_t*)queries_grad.data_ptr<at::Half>()
    );

    CHECK_LAST_CUDA_ERROR();

    launch_attention_backward_kernel_b<128, 64>(
        batch_size, seq_len, num_features,
        (half_t*)queries.data_ptr<at::Half>(),
        (half_t*)keys.data_ptr<at::Half>(),
        (half_t*)values.data_ptr<at::Half>(),
        (half_t*)output_grad.data_ptr<at::Half>(),
        (half_t*)scores_max.data_ptr<at::Half>(),
        (half_t*)scores_sum.data_ptr<at::Half>(),
        (half_t*)scores_grad_sum.data_ptr<at::Half>(),
        (half_t*)keys_grad.data_ptr<at::Half>(),
        (half_t*)values_grad.data_ptr<at::Half>()
    );

    CHECK_LAST_CUDA_ERROR();

    return { queries_grad, keys_grad, values_grad };
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
          py::arg("inference") = false,
          "Fused multihead attention forward");

    m.def("attention_backward",
          &attention_backward,
          py::arg("head_dim"),
          py::arg("chunk_size"),
          py::arg("queries"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("output_grad"),
          py::arg("score_max"),
          py::arg("score_sum"),
          "Fused multihead attention backwards");
}
