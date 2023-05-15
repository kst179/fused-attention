#include <iostream>
#include <cuda.h>
#include "fused_attn.cuh"

__global__
void fill_values(uint32_t size, uint32_t seed, half_t* data) {
    uint32_t s = seed;
    
    for (int i = 0; i < size; ++i) {
        s = (s * 32767) % 65521; // some prime numbers for simple prnd

        data[i] = __float2half((float)s / (1u << 16) * 10 - 5);
    }
}

__global__
void print_matrix(uint32_t height, uint32_t width, half_t* data, uint32_t ldm) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%10.2f ", __half2float(data[i * ldm + j]));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    constexpr uint32_t head_dim = 64;
    constexpr uint32_t chunk_size = 128;

    int batch_size = 1;
    int seq_len = 2 * chunk_size;
    int num_features = 2 * head_dim;

    int size = batch_size * seq_len * num_features;
    int size_bytes = size * sizeof(half_t);

    half_t* queries;
    half_t* keys;
    half_t* values;
    half_t* output;

    CHECK_CUDA_ERROR(cudaMalloc(&queries, size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&keys, size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&values, size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&output, size_bytes));

    fill_values<<<1, 1>>>(size, 1, queries);
    fill_values<<<1, 1>>>(size, 2, keys);
    fill_values<<<1, 1>>>(size, 3, values);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    print_matrix<<<1, 1>>>(16, 16, queries, num_features);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    launch_attention_kernel<head_dim, chunk_size>(batch_size, seq_len, num_features,
                                                  queries, keys, values, nullptr, output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();      

    print_matrix<<<1, 1>>>(16, 16, output, num_features);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return 0;
}