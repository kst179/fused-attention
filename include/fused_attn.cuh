#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

using half_t = __half;
using half2_t = __half2;

[[maybe_unused]]
constexpr uint32_t warp_size = 32;


#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCudaError(T err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)
void checkLastCudaError(char const* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


/*
 * Loads matrix from global to shared memory (row-wise)
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_load_chunk(
    const half_t* __restrict__ src,
    half_t* __restrict__ dst,
    uint32_t lds,
    uint32_t ldd
) {
    constexpr uint32_t elements_per_storage = 8; // 8 half_t == 1x uint4 == 128 bit
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
    
    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        *(uint4*)&dst[(offset + row) * ldd + col] = *(uint4*)&src[(offset + row) * lds + col];
    }
}

/*
 * Stores matrix chunk from shared to global memory (row-wise)
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_store_chunk(
    const half_t* __restrict__ src,
    half_t* __restrict__ dst,
    uint32_t lds,
    uint32_t ldd
) {
    constexpr uint32_t elements_per_storage = 8;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
    
    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        *(uint4*)&dst[(offset + row) * ldd + col] = *(uint4*)&src[(offset + row) * lds + col];
    }
}

/*
 * Stores matrix chunk to global memory and performs last-mile division 
 * of the softmax aggregated numerator which is sum(exp(QK.T)V) and denominator which is sum(exp(QK.T))
 * fused operation reduces some loads/stores between shared memory and registers
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_divide_and_store_chunk(
    const half_t* __restrict__ numer,
    const half_t* __restrict__ denom,
    half_t* __restrict__ dst,
    uint32_t lds,
    uint32_t ldd
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;
    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
    
    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        *(half2_t*)&dst[(offset + row) * ldd + col] = __h2div(
            *(half2_t*)&numer[(offset + row) * lds + col],
            __half2half2(denom[offset + row])
        );
    }
}

/*
 * Calculates classic gemm: C = alpha * A * B + beta * C (but with beta always equals to 0) 
 * here A is of size (m, k) and B of size (k, n)
 * Can be also used for C = alpha * A * B^T calculation (then B should be of size (n, k))
 * Multiplication by alpha can be compile-time suppressed by setting tparam alpha_is_one = True
 */
template < uint32_t m, uint32_t n, uint32_t k, int n_warps, 
           bool transpose_b=false, bool alpha_is_one=true, bool preloaded_a_frags=false >
__device__
void threadblock_gemm(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half_t, nvcuda::wmma::row_major> mat_a_frags[m / 16][k / 16],
    const half_t* __restrict__ mat_a,
    const half_t* __restrict__ mat_b,
    half_t* __restrict__ mat_c,
    uint32_t lda,
    uint32_t ldb,
    uint32_t ldc,
    __half alpha = __float2half(1.0f)
) {
    constexpr uint32_t frag_cols_per_warp = n / (16 * n_warps); // num of columns processed by single warp (in terms of fragments)
    constexpr uint32_t frag_rows = m / 16; // num of fragments in rows in output matrix
    constexpr uint32_t frag_cols = n / 16; // num of fragments in cols in output matrix
    constexpr uint32_t frag_dims = k / 16; // size of common dimention of a and b matrices (in fragments)

    static_assert(n_warps * frag_cols_per_warp == frag_cols);
    static_assert(n % (n_warps * 16) == 0);

    using namespace nvcuda;
    using mat_b_order = std::conditional_t<transpose_b, wmma::col_major, wmma::row_major>;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, mat_b_order> frag_b[frag_dims][frag_cols_per_warp];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half_t> frag_acc;

    const uint32_t warp_idx = threadIdx.y;

    // Load mat b to fragments distributed by cols between warps 
    // if n_warps is smaller than fragments' columns then one warp works with several columns
    #pragma unroll
    for (uint32_t frag_dim = 0; frag_dim < frag_dims; ++frag_dim) {
        #pragma unroll
        for (uint32_t frag_col_offset = 0; frag_col_offset < frag_cols_per_warp; frag_col_offset++) {
            const uint32_t frag_col = warp_idx * frag_cols_per_warp + frag_col_offset; 
            
            if (transpose_b) {
                wmma::load_matrix_sync(frag_b[frag_dim][frag_col_offset],
                                       &mat_b[16 * (frag_dim + frag_col * ldb)], ldb);
            } else {
                wmma::load_matrix_sync(frag_b[frag_dim][frag_col_offset],
                                       &mat_b[16 * (frag_dim * ldb + frag_col)], ldb);
            }
        }
    }

    // Iter trough rows
    #pragma unroll
    for (uint32_t frag_row = 0; frag_row < frag_rows; ++frag_row) {

        // Iter trough columns of single warp
        #pragma unroll
        for (uint32_t frag_col_offset = 0; frag_col_offset < frag_cols_per_warp; ++frag_col_offset) {
            const uint32_t frag_col = warp_idx * frag_cols_per_warp + frag_col_offset;

            wmma::fill_fragment(frag_acc, __float2half(0.0f));
            
            #pragma unroll
            for (uint32_t frag_dim = 0; frag_dim < frag_dims; ++frag_dim) {
                if (preloaded_a_frags) {
                    wmma::mma_sync(frag_acc, mat_a_frags[frag_row][frag_dim], frag_b[frag_dim][frag_col_offset], frag_acc);
                } else {
                    wmma::load_matrix_sync(frag_a, &mat_a[16 * (frag_row * lda + frag_dim)], lda);
                    wmma::mma_sync(frag_acc, frag_a, frag_b[frag_dim][frag_col_offset], frag_acc);    
                }
            }

            // multiply result by alpha
            if (!alpha_is_one) {
                for(int t = 0; t < frag_acc.num_elements; t++) {
                    frag_acc.x[t] = __hmul(frag_acc.x[t], alpha);
                }
            }

            wmma::store_matrix_sync(&mat_c[16 * (frag_row * ldc + frag_col)], frag_acc, ldc, wmma::mem_row_major);
        }

    }
}

/*
 * Calculates rowwise sums of matrix
 * Uses additional auxillary matrix which should be of size (height, 16) to store fragmens with rowwise sums into shared memory
 */
template <uint32_t height, uint32_t width, uint32_t n_warps, bool copy_to_vec=true>
__device__
void threadblock_row_sum(
    const half_t* __restrict__ mat,
    half_t* __restrict__ vec,
    half_t* __restrict__ aux,
    uint32_t ldm,
    uint32_t ldm_aux
) {
    constexpr uint32_t frag_rows = height / 16;
    constexpr uint32_t frag_cols = width / 16;

    static_assert(frag_rows % n_warps == 0);        // can distribute rows between warps (each warp calculates rowwise sum in one row of fragments)
    static_assert(height <= n_warps * warp_size);   // can copy results to vec in single iteration

    using namespace nvcuda;

    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;
    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * 2;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half_t> frag_acc;

    wmma::fill_fragment(frag_b, __float2half(1.0f));

    #pragma unroll
    for (uint32_t frag_row_offset = 0; frag_row_offset < frag_rows; frag_row_offset += n_warps) {
        const uint32_t frag_row = frag_row_offset + warp_idx;
        wmma::fill_fragment(frag_acc, __float2half(0.0f));

        #pragma unroll
        for(uint32_t frag_col = 0; frag_col < frag_cols; ++frag_col) {
            wmma::load_matrix_sync(frag_a, &mat[16 * (frag_row * ldm + frag_col)], ldm);
            wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }

        // Store in transposed mem to obtain all row-wise sums in coaleced vector in aux memory
        wmma::store_matrix_sync(&aux[16 * frag_row], frag_acc, ldm_aux, wmma::mem_col_major);
    }

    if (copy_to_vec) {
        __syncthreads();

        if (storage_idx < height) {
            *(half2_t*)&vec[storage_idx] = *(half2_t*)&aux[storage_idx];
        }
    }
}

template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__ 
void threadblock_fill_value(
    half_t* __restrict__ mat,
    uint32_t ldm,
    const half_t val
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        const uint32_t idx = (offset + row) * ldm + col;

        *(half2_t*)&mat[idx] = __half2half2(val);
    }
}

template <uint32_t size, uint32_t n_warps>
__device__ 
void threadblock_vec_fill_value(
    half_t* __restrict__ vec,
    const half_t val
) {
    constexpr uint32_t elements_per_storage = 2;

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    if (storage_idx < size) {
        *(half2_t*)&vec[storage_idx] = __half2half2(val);
    }
}

/*
 * Calculates A = A + B elementwise
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__ 
void threadblock_ewise_sum(
    half_t* __restrict__ mat_a,
    const half_t* __restrict__ mat_b,
    uint32_t lda,
    uint32_t ldb
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        const uint32_t idx_a = (offset + row) * lda + col;
        const uint32_t idx_b = (offset + row) * ldb + col;

        *(half2_t*)&mat_a[idx_a] = __hadd2(*(half2_t*)&mat_a[idx_a], *(half2_t*)&mat_b[idx_b]);
    }
}

/*
 * Calculates rowwise maximum of matrix and stores it to given vector
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_row_max(
    const half_t* __restrict__ mat,
    half_t* __restrict__ vec,
    half_t* __restrict__ aux,
    uint32_t ldm,
    uint32_t ldm_aux
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t threads_per_row = n_threads / height;
    constexpr uint32_t storage_per_thread = width * height / (n_threads * elements_per_storage);

    static_assert(n_threads % height == 0);                               // can equally distribute threads between rows
    static_assert(width % (threads_per_row * elements_per_storage) == 0); // can distribute elements between all threads in the same row

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t thread_idx = warp_idx * warp_size + lane_idx;
    const uint32_t storage_idx = thread_idx * storage_per_thread * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;
    const uint32_t col_aux = col / storage_per_thread;

    half2_t threadwise_val = *(half2_t*)&mat[row * ldm + col];

    #pragma unroll
    for (uint32_t i = 1; i < storage_per_thread; ++i) {
        threadwise_val = __hmax2(threadwise_val, *(half2_t*)&mat[row * ldm + col + i * elements_per_storage]);
    }

    if (threads_per_row == 1) {
        vec[row] = __hmax(threadwise_val.x, threadwise_val.y);
        return;
    }

    *(half2_t*)&aux[row * ldm_aux + col_aux] = threadwise_val;
    
    __syncthreads();

    if (thread_idx < height) {
        threadwise_val = *(half2_t*)&aux[thread_idx * ldm_aux];

        #pragma unroll
        for (uint32_t i = 1; i < threads_per_row; ++i) {
            threadwise_val = __hmax2(threadwise_val, *(half2_t*)&aux[thread_idx * ldm_aux + i * elements_per_storage]);
        }
    
        vec[thread_idx] = __hmax(threadwise_val.x, threadwise_val.y);
    }
}

/*
 * Calculates A = exp(A - b[:, None]), where A - matrix, b - column-vector
 * Used for stable exp calculation in softmax
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_row_broadcast_diff_and_exp(
    half_t* __restrict__ mat,
    const half_t* __restrict__ vec,
    uint32_t ldm
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        const uint32_t idx = (offset + row) * ldm + col;
        
        *(half2_t*)&mat[idx] = h2exp(__hsub2(*(half2_t*)&mat[idx], __half2half2(vec[offset + row])));
    }
}

/*
 * Calculates: 
 *      max_a' = max(max_a, max_b)
 *      numer_a' = numer_a * exp(max_a - max_a') + numer_b * exp(max_b - max_a')
 *      denom_a' = denom_a * exp(max_a - max_a') + denom_b * exp(max_b - max_a')
 * Which is casual sum: 
 *      numer_A = numer_A + numer_B 
 *      denom_a = denom_a + denom_b
 * but all numers represented as x' = exp(max_x) * x and exponentials are not caclulated due to computation errors
 */
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_aggregate_softmax(
    half_t* __restrict__ numer_a,
    half_t* __restrict__ denom_a,
    half_t* __restrict__ max_a,
    const half_t* __restrict__ numer_b,
    const half_t* __restrict__ denom_b,
    half_t* __restrict__ max_b,
    uint32_t lda,
    uint32_t ldb,
    half_t* __restrict__ aux
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t rows_per_iter = warp_size * n_warps * elements_per_storage / width;

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    if (storage_idx < height) {
        half2_t* __restrict__ a2_ptr = (half2_t*)&max_a[storage_idx];
        half2_t* __restrict__ b2_ptr = (half2_t*)&max_b[storage_idx];
        half2_t* __restrict__ max_ab2_ptr = (half2_t*)&aux[storage_idx];
        
        *max_ab2_ptr = __hmax2(*a2_ptr, *b2_ptr);
        *a2_ptr = h2exp(__hsub2(*a2_ptr, *max_ab2_ptr));
        *b2_ptr = h2exp(__hsub2(*b2_ptr, *max_ab2_ptr));

        *(half2_t*)&denom_a[storage_idx] = __hadd2(__hmul2(*a2_ptr, *(half2_t*)&denom_a[storage_idx]),
                                                   __hmul2(*b2_ptr, *(half2_t*)&denom_b[storage_idx]));
    }

    __syncthreads();

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        const uint32_t idx_a = (offset + row) * lda + col;
        const uint32_t idx_b = (offset + row) * ldb + col;


        *(half2_t*)&numer_a[idx_a] = __hadd2(__hmul2(__half2half2(max_a[offset + row]), *(half2_t*)&numer_a[idx_a]),
                                             __hmul2(__half2half2(max_b[offset + row]), *(half2_t*)&numer_b[idx_b]));
    }

    __syncthreads();
    
    if (storage_idx < height) {
        *(half2_t*)&max_a[storage_idx] = *(half2_t*)&aux[storage_idx];
    }
}

/*
 * Usage: distribute_shared_mem<T, size_a, size_b, size_c>(shmem, &a, &b, &c);
 * Where shmem is pointer to shared memory, and a, b, c are pointers to arrays of corresponding sizes
 */
template <typename T, int size>
__device__
inline void distribute_shared_mem(T* mem_ptr, T** array_ptr) {
    *array_ptr = mem_ptr;
}

template <typename T, int size, int... sizes, typename... Args>
__device__
inline void distribute_shared_mem(T* mem_ptr, T** array_ptr, Args... array_ptrs) {
    *array_ptr = mem_ptr;
    distribute_shared_mem<T, sizes...>(mem_ptr + size, array_ptrs...);
}

template <uint32_t height, uint32_t width>
__device__
void print_matrix(half_t* mat, uint32_t ldm) {
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                printf("%8.4f ", __half2float(mat[i * ldm + j]));
            }
            printf("\n");
        }
        printf("\n\n");
    }

    __syncthreads();
}



/*
chunk_size = 64 (or 128?)
head_size = 128

K = L / chunk_size
N_heads = F / head_size

queries     (batch_size, seq_len, num_features)
keys        -//-
values      -//-
attn_mask   (batch_size, seq_len, seq_len)

                               | elements  -> chunks (chunk_size, head_size)
S = softmax(Q @ K.T / sqrt(d)) | (B, L, L) -> (B, K, K)
Y = S @ V                      | (B, L, F) -> (B, K, N_heads)

for each batch = 1 .. n_batches:        // can be parallelized, gridDim.x
    for each head = 1 .. n_heads:       // can be parallelized, gridDim.y
        for each q_row = 1 .. K:        // can be parallelized, gridDim.z

            query_chunk = Q[q_row, head]    // Load query chunk to shmem

            for each kv_row = 1 .. K: // done seuqential inside threadblock
                // calculate score:
                // score[q_row, kv_row] = Q[q_row, head] @ K[kv_row, head].T / sqrt(head_dim)
                score_chunk = query_chunk @ K[kv_row, head].T / sqrt(head_dim)

                // score_max[q_row] = max_row(score[q_row, kv_row])
                score_max = max(score_chunk, dim=1)

                // score[q_row, kv_row] = exp(score[q_row, kv_row] - score_max[q_row])
                score_chunk = exp(score_chunk - score_max.view(chunk_size, 1))
                
                numer_local = score_chunk @ V[kv_row, head]
                denom_local = sum(score_chunk, dim=1)

                tmp_max = maximum(score_max, score_max_old)

                numer = exp(score_max - tmp_max) * numer + exp(score_max_local - tmp_max) * numer_local
                denom = exp(score_max - tmp_max) * denom + exp(score_max_local - tmp_max) * denom_local
                score_max = tmp_max

            // Y[q_row, head] = exp(logits_numer[q_row, head] - logits_denom[q_row, head])
            Y[q_row, head] = exp(logits_numer - logits_denom.view(chunk_size, 1))
*/
template <
    uint32_t head_dim,
    uint32_t chunk_size,
    uint32_t n_warps
>
__global__
void attention_kernel(
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_features,
    const half_t* __restrict__ queries,
    const half_t* __restrict__ keys,
    const half_t* __restrict__ values,
    const half_t* __restrict__ mask,
    half_t* __restrict__ output
) {
    using namespace nvcuda;

    constexpr uint32_t mat_skew = 8;
    // constexpr uint32_t common_ldm = head_dim + mat_skew;
    constexpr uint32_t reduce_max_ldm = 2 * warp_size * n_warps / chunk_size;

    constexpr uint32_t chunk_frags = chunk_size / 16;
    constexpr uint32_t head_frags = head_dim / 16;

    constexpr uint32_t max_size = chunk_size < head_dim ? head_dim : chunk_size;

    // static_assert(chunk_size <= head_dim);
    static_assert(chunk_size <= warp_size * n_warps);                     // Column operations should be done in one iter (or less)
    static_assert(chunk_size * head_dim % (warp_size * n_warps) == 0);    // Matrix operations should be done in several iters
    static_assert(chunk_size * chunk_size % (warp_size * n_warps) == 0);
    static_assert(reduce_max_ldm <= 16);

    half_t* queries_chunk;      // matrix (chunk_size, head_dim)
    half_t* scores_chunk;       // matrix (chunk_size, chunk_size)
    half_t* numer_local;        // matrix (chunk_size, head_dim)
    half_t* numer;              // matrix (chunk_size, head_dim)
    half_t* aux_mem;            // matrix (chunk_size, 16)

    constexpr uint32_t numer_ldm = head_dim + mat_skew;
    constexpr uint32_t queries_chunk_ldm = numer_ldm;
    constexpr uint32_t numer_local_ldm = max_size + mat_skew;
    constexpr uint32_t scores_chunk_ldm = numer_local_ldm;
    constexpr uint32_t reduce_sum_ldm = chunk_size + mat_skew;

    half_t* scores_max_local;   // vector (chunk_size,)
    half_t* denom_local;        // vector (chunk_size,)
    half_t* denom;              // vector (chunk_size,)
    half_t* scores_max;         // vector (chunk_size,)

    extern __shared__ half_t shmem[];

    distribute_shared_mem<half_t,
        chunk_size * numer_ldm,                      // numer / queries_chunk
        (chunk_size + 16) * numer_local_ldm,         // numer_local / scores_chunk
        16 * reduce_sum_ldm,                         // aux_mem
        chunk_size,                                  // scores_max_local
        chunk_size,                                  // denom
        chunk_size,                                  // denom_local
        chunk_size                                   // scores_max
    >(
        shmem,
        &numer,
        &numer_local,
        &aux_mem,
        &scores_max_local,
        &denom,
        &denom_local,
        &scores_max
    );

    queries_chunk = numer;
    scores_chunk = &numer_local[16 * numer_local_ldm];

    const uint32_t num_chanks = seq_len / chunk_size;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t q_row_chunk = blockIdx.y;
    const uint32_t head_col_chunk = blockIdx.z;

    const uint32_t batch_offset = batch_idx * seq_len * num_features;
    const uint32_t batch_offset_mask = batch_idx * seq_len * seq_len;

    // row / col in the matrix of fixed batch_idx (seq_len, num_features)
    const uint32_t q_row = q_row_chunk * chunk_size;
    const uint32_t head_col = head_col_chunk * head_dim;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> query_frags[chunk_frags][head_frags];

    // Load query chunk into shared memory
    threadblock_load_chunk<chunk_size, head_dim, n_warps>(
        /*src = */ &queries[batch_offset + q_row * num_features + head_col],
        /*dst = */ queries_chunk,
        /*lds = */ num_features,
        /*ldd = */ queries_chunk_ldm
    );

    __syncthreads();

    // Move query chunk into warps' fragments, query_chunk is freed here and can be used later as numer
    #pragma unroll
    for (uint32_t row_frag = 0; row_frag < chunk_frags; ++row_frag) {
        #pragma unroll
        for (uint32_t col_frag = 0; col_frag < head_frags; ++col_frag) {
            wmma::load_matrix_sync(query_frags[row_frag][col_frag], &queries_chunk[16 * (row_frag * queries_chunk_ldm + col_frag)], queries_chunk_ldm);
        }
    }

    #pragma unroll(1)
    for (uint32_t kv_row_chunk = 0; kv_row_chunk < num_chanks; kv_row_chunk++) {
        const uint32_t kv_row = kv_row_chunk * chunk_size;
        
        // scores_chunk = queries_chunk @ keys_chunk.T / sqrt(head_dim)
        threadblock_gemm< /*m=*/chunk_size, /*n=*/chunk_size, /*k=*/head_dim, n_warps,
                          /*transpose_b=*/true, /*alpha_is_one=*/false, /*preloaded_a_frags=*/true >(
            /*mat_a_frags=*/ query_frags,
            /*mat_a =*/ nullptr,
            /*mat_b =*/ &keys[batch_offset + kv_row * num_features + head_col],
            /*mat_c =*/ scores_chunk,
            /*lda =*/ 0,
            /*ldb =*/ num_features,
            /*ldc =*/ scores_chunk_ldm,
            /*alpha =*/ __float2half(rsqrtf(head_dim))
        );

        __syncthreads();

        if (kv_row_chunk == 0) {
            // First iter: write directly to numer, denom, scores_max, no aggregation

            if (mask != nullptr) {
                threadblock_ewise_sum<chunk_size, chunk_size, n_warps>(
                    /*mat_a =*/ scores_chunk,
                    /*mat_b =*/ &mask[batch_offset_mask + q_row * seq_len + kv_row],
                    /*lda =*/ scores_chunk_ldm,
                    /*ldb =*/ seq_len
                );
            }

            // scores_max = max(scores_chunk, dim=1)
            threadblock_row_max<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm
            );

            __syncthreads();

            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*ldm =*/ scores_chunk_ldm
            );

            __syncthreads();

            // denom = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size, chunk_size, n_warps>(
                /*mat = */ scores_chunk,
                /*vec = */ denom,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm
            );

            __syncthreads();

            // numer = scores_chunk @ values_chunk
            threadblock_gemm< /*m=*/chunk_size, /*n=*/head_dim, /*k=*/chunk_size, n_warps, 
                              /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset + head_col],
                /*mat_c = */ numer,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ num_features,
                /*ldc = */ numer_ldm
            );

        } else {
            // Second and further iterations, write to temprorary numer_local, denom_local, scores_max_local, then aggregate with prev values

            // scores_max = max(scores_chunk, dim=1)
            threadblock_row_max<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm
            );

            __syncthreads();

            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*ldm =*/ scores_chunk_ldm
            );

            __syncthreads();

            // denom_local = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size, chunk_size, n_warps, /*copy_to_vec=*/true>(
                /*mat = */ scores_chunk,
                /*vec = */ denom_local,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm
            );
             
            __syncthreads();

            // numer_local = scores_chunk @ values_chunk
            threadblock_gemm< /*m=*/chunk_size, /*n=*/head_dim, /*k=*/chunk_size, n_warps, 
                              /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset + kv_row * num_features + head_col],
                /*mat_c = */ numer_local,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ num_features,
                /*ldc = */ numer_local_ldm
            );

            __syncthreads();

            threadblock_aggregate_softmax<chunk_size, head_dim, n_warps>(
                /*numer_a = */ numer,
                /*denom_a = */ denom,
                /*max_a = */ scores_max,
                /*numer_b = */ numer_local,
                /*denom_b = */ denom_local,
                /*max_b = */ scores_max_local,
                /*lda =*/ numer_ldm,
                /*ldb =*/ numer_local_ldm,
                /*aux =*/ aux_mem
            );
        }
    }

    __syncthreads();

    threadblock_divide_and_store_chunk<chunk_size, head_dim, n_warps>(
        /*numer = */ numer,
        /*denom = */ denom,
        /*dst = */ &output[batch_offset + q_row * num_features + head_col],
        /*lds = */ numer_ldm,
        /*ldd = */ num_features
    );
}

template <uint32_t head_dim, uint32_t chunk_size>
void launch_attention_kernel(
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_features,
    const half_t* queries,
    const half_t* keys,
    const half_t* values,
    const half_t* mask,
    half_t* output,
    bool syncronize = false
) {
    constexpr uint32_t n_warps = chunk_size < head_dim ? chunk_size / 16 : head_dim / 16;
    constexpr uint32_t max_size = chunk_size > head_dim ? chunk_size : head_dim;
    constexpr uint32_t shared_mem_size = ( chunk_size * (head_dim + 8) + (chunk_size + 16) * (max_size + 8) + 
                                           16 * (chunk_size + 8) + 4 * chunk_size ) * sizeof(half_t);

    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        attention_kernel<head_dim, chunk_size, n_warps>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

    // Call attention kernel
    dim3 threads(warp_size, n_warps, 1);
    dim3 blocks(batch_size, seq_len / chunk_size, num_features / head_dim);

    attention_kernel<head_dim, chunk_size, n_warps><<<blocks, threads, shared_mem_size>>>(
        batch_size, seq_len, num_features, 
        queries, keys, values, mask, output
    );

    if (syncronize) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_LAST_CUDA_ERROR();        
    }
}