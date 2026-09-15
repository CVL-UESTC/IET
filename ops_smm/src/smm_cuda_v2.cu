#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ float smm_to_float(scalar_t value);

template <>
__device__ __forceinline__ float smm_to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float smm_to_float<__half>(__half value) {
    return __half2float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t smm_from_float(float value);

template <>
__device__ __forceinline__ float smm_from_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half smm_from_float<__half>(float value) {
    return __float2half(value);
}

template <typename scalar_t>
__global__ void SMM_QmK_forward_v2_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K_mat,
    const int* __restrict__ index,
    scalar_t* __restrict__ logits,
    int N,
    int candidates,
    int head_dim,
    int key_rows,
    float invalid_value) {
    const int batch = blockIdx.y;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int base = (batch * N + row) * candidates;
    const scalar_t* q_ptr = Q + (batch * N + row) * head_dim;

    for (int candidate = warp; candidate < candidates; candidate += warps) {
        const int key_row = index[base + candidate];
        float value = 0.0f;
        if (key_row >= 0 && key_row < key_rows) {
            const scalar_t* k_ptr = K_mat + (batch * key_rows + key_row) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                value += smm_to_float(q_ptr[d]) * smm_to_float(k_ptr[d]);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
        }
        if (lane == 0) {
            logits[base + candidate] = smm_from_float<scalar_t>(
                key_row >= 0 && key_row < key_rows ? value : invalid_value);
        }
    }
}

template <typename scalar_t>
__global__ void SMM_AmV_forward_v2_kernel(
    const scalar_t* __restrict__ attention,
    const scalar_t* __restrict__ V,
    const int* __restrict__ index,
    scalar_t* __restrict__ output,
    int N,
    int candidates,
    int head_dim,
    int value_rows) {
    const int batch = blockIdx.y;
    const int row = blockIdx.x;
    const int sparse_base = (batch * N + row) * candidates;
    const int output_base = (batch * N + row) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (int candidate = 0; candidate < candidates; ++candidate) {
            const int value_row = index[sparse_base + candidate];
            if (value_row >= 0 && value_row < value_rows) {
                value += smm_to_float(attention[sparse_base + candidate]) *
                         smm_to_float(V[(batch * value_rows + value_row) * head_dim + d]);
            }
        }
        output[output_base + d] = smm_from_float<scalar_t>(value);
    }
}

at::Tensor SMM_QmK_forward_cuda_v2(
    const at::Tensor &Q, const at::Tensor &K_mat, const at::Tensor &index) {
    TORCH_CHECK(Q.is_cuda() && K_mat.is_cuda() && index.is_cuda(), "Q, K, and index must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K_mat.is_contiguous() && index.is_contiguous(), "Q, K, and index must be contiguous");
    TORCH_CHECK(Q.dim() == 3 && K_mat.dim() == 3 && index.dim() == 3, "Q, K, and index must be rank-3 tensors");
    TORCH_CHECK(Q.scalar_type() == K_mat.scalar_type(), "Q and K must have the same dtype");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(Q.size(0) == K_mat.size(0) && Q.size(0) == index.size(0), "batch dimensions must match");
    TORCH_CHECK(Q.size(1) == index.size(1), "Q rows and index rows must match");
    TORCH_CHECK(Q.size(2) == K_mat.size(2), "Q and K head dimensions must match");

    const int batch = Q.size(0);
    const int N = Q.size(1);
    const int candidates = index.size(2);
    const int head_dim = Q.size(2);
    const int key_rows = K_mat.size(1);
    auto logits = at::empty({batch, N, candidates}, Q.options());
    const dim3 grid(N, batch);
    constexpr int threads = 256;

    if (Q.scalar_type() == at::kFloat) {
        SMM_QmK_forward_v2_kernel<float><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            Q.data_ptr<float>(), K_mat.data_ptr<float>(), index.data_ptr<int>(), logits.data_ptr<float>(),
            N, candidates, head_dim, key_rows, -INFINITY);
    } else {
        TORCH_CHECK(Q.scalar_type() == at::kHalf, "optimized QmK supports float32 and float16 only");
        SMM_QmK_forward_v2_kernel<__half><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K_mat.data_ptr<at::Half>()),
            index.data_ptr<int>(), reinterpret_cast<__half*>(logits.data_ptr<at::Half>()),
            N, candidates, head_dim, key_rows, -INFINITY);
    }
    return logits;
}

at::Tensor SMM_AmV_forward_cuda_v2(
    const at::Tensor &attention, const at::Tensor &V, const at::Tensor &index) {
    TORCH_CHECK(attention.is_cuda() && V.is_cuda() && index.is_cuda(), "attention, V, and index must be CUDA tensors");
    TORCH_CHECK(attention.is_contiguous() && V.is_contiguous() && index.is_contiguous(), "attention, V, and index must be contiguous");
    TORCH_CHECK(attention.dim() == 3 && V.dim() == 3 && index.dim() == 3, "attention, V, and index must be rank-3 tensors");
    TORCH_CHECK(attention.scalar_type() == V.scalar_type(), "attention and V must have the same dtype");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(attention.sizes() == index.sizes(), "attention and index shapes must match");
    TORCH_CHECK(attention.size(0) == V.size(0), "batch dimensions must match");

    const int batch = attention.size(0);
    const int N = attention.size(1);
    const int candidates = attention.size(2);
    const int value_rows = V.size(1);
    const int head_dim = V.size(2);
    auto output = at::empty({batch, N, head_dim}, V.options());
    const dim3 grid(N, batch);
    constexpr int threads = 64;

    if (attention.scalar_type() == at::kFloat) {
        SMM_AmV_forward_v2_kernel<float><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            attention.data_ptr<float>(), V.data_ptr<float>(), index.data_ptr<int>(), output.data_ptr<float>(),
            N, candidates, head_dim, value_rows);
    } else {
        TORCH_CHECK(attention.scalar_type() == at::kHalf, "optimized AmV supports float32 and float16 only");
        SMM_AmV_forward_v2_kernel<__half><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(attention.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            index.data_ptr<int>(), reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            N, candidates, head_dim, value_rows);
    }
    return output;
}

// Scatter a candidate-weighted dense row into the indexed dense matrix.  A
// warp owns one sparse candidate so its lanes update consecutive head
// dimensions.  Accumulation is always FP32, including for FP16 inputs.
template <typename scalar_t>
__global__ void SMM_scatter_rows_backward_v2_kernel(
    const scalar_t* __restrict__ coefficients,
    const scalar_t* __restrict__ dense_rows,
    const int* __restrict__ index,
    float* __restrict__ grad_indexed_rows,
    int N,
    int candidates,
    int head_dim,
    int indexed_rows) {
    const int batch = blockIdx.y;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int sparse_base = (batch * N + row) * candidates;
    const scalar_t* dense_row = dense_rows + (batch * N + row) * head_dim;

    for (int candidate = warp; candidate < candidates; candidate += warps) {
        const int indexed_row = index[sparse_base + candidate];
        if (indexed_row < 0 || indexed_row >= indexed_rows) {
            continue;
        }
        const float coefficient = smm_to_float(coefficients[sparse_base + candidate]);
        float* grad_row = grad_indexed_rows +
            (batch * indexed_rows + indexed_row) * head_dim;
        for (int d = lane; d < head_dim; d += 32) {
            atomicAdd(grad_row + d, coefficient * smm_to_float(dense_row[d]));
        }
    }
}

std::vector<at::Tensor> SMM_QmK_backward_cuda_v2(
    const at::Tensor &grad_output,
    const at::Tensor &Q,
    const at::Tensor &K_mat,
    const at::Tensor &index) {
    TORCH_CHECK(grad_output.is_cuda() && Q.is_cuda() && K_mat.is_cuda() && index.is_cuda(),
                "grad_output, Q, K, and index must be CUDA tensors");
    TORCH_CHECK(grad_output.is_contiguous() && Q.is_contiguous() && K_mat.is_contiguous() && index.is_contiguous(),
                "grad_output, Q, K, and index must be contiguous");
    TORCH_CHECK(grad_output.dim() == 3 && Q.dim() == 3 && K_mat.dim() == 3 && index.dim() == 3,
                "grad_output, Q, K, and index must be rank-3 tensors");
    TORCH_CHECK(grad_output.scalar_type() == Q.scalar_type() && Q.scalar_type() == K_mat.scalar_type(),
                "grad_output, Q, and K must have the same dtype");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(grad_output.sizes() == index.sizes(), "grad_output and index shapes must match");
    TORCH_CHECK(Q.size(0) == K_mat.size(0) && Q.size(0) == index.size(0), "batch dimensions must match");
    TORCH_CHECK(Q.size(1) == index.size(1), "Q rows and index rows must match");
    TORCH_CHECK(Q.size(2) == K_mat.size(2), "Q and K head dimensions must match");

    const int batch = Q.size(0);
    const int N = Q.size(1);
    const int candidates = index.size(2);
    const int head_dim = Q.size(2);
    const int key_rows = K_mat.size(1);
    auto grad_Q = at::empty_like(Q);
    auto grad_K_accum = at::zeros({batch, key_rows, head_dim}, Q.options().dtype(torch::kFloat32));
    const dim3 grid(N, batch);
    constexpr int av_threads = 64;
    constexpr int scatter_threads = 256;

    if (Q.scalar_type() == at::kFloat) {
        SMM_AmV_forward_v2_kernel<float><<<grid, av_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.data_ptr<float>(), K_mat.data_ptr<float>(), index.data_ptr<int>(), grad_Q.data_ptr<float>(),
            N, candidates, head_dim, key_rows);
        SMM_scatter_rows_backward_v2_kernel<float><<<grid, scatter_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.data_ptr<float>(), Q.data_ptr<float>(), index.data_ptr<int>(), grad_K_accum.data_ptr<float>(),
            N, candidates, head_dim, key_rows);
    } else {
        TORCH_CHECK(Q.scalar_type() == at::kHalf, "optimized QmK backward supports float32 and float16 only");
        SMM_AmV_forward_v2_kernel<__half><<<grid, av_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K_mat.data_ptr<at::Half>()), index.data_ptr<int>(),
            reinterpret_cast<__half*>(grad_Q.data_ptr<at::Half>()), N, candidates, head_dim, key_rows);
        SMM_scatter_rows_backward_v2_kernel<__half><<<grid, scatter_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), index.data_ptr<int>(),
            grad_K_accum.data_ptr<float>(), N, candidates, head_dim, key_rows);
    }
    auto grad_K = Q.scalar_type() == at::kFloat ? grad_K_accum : grad_K_accum.to(Q.scalar_type());
    return {grad_Q, grad_K};
}

std::vector<at::Tensor> SMM_AmV_backward_cuda_v2(
    const at::Tensor &grad_output,
    const at::Tensor &attention,
    const at::Tensor &V,
    const at::Tensor &index) {
    TORCH_CHECK(grad_output.is_cuda() && attention.is_cuda() && V.is_cuda() && index.is_cuda(),
                "grad_output, attention, V, and index must be CUDA tensors");
    TORCH_CHECK(grad_output.is_contiguous() && attention.is_contiguous() && V.is_contiguous() && index.is_contiguous(),
                "grad_output, attention, V, and index must be contiguous");
    TORCH_CHECK(grad_output.dim() == 3 && attention.dim() == 3 && V.dim() == 3 && index.dim() == 3,
                "grad_output, attention, V, and index must be rank-3 tensors");
    TORCH_CHECK(grad_output.scalar_type() == attention.scalar_type() && attention.scalar_type() == V.scalar_type(),
                "grad_output, attention, and V must have the same dtype");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(attention.sizes() == index.sizes(), "attention and index shapes must match");
    TORCH_CHECK(attention.size(0) == V.size(0) && attention.size(0) == grad_output.size(0),
                "batch dimensions must match");
    TORCH_CHECK(attention.size(1) == grad_output.size(1), "row dimensions must match");
    TORCH_CHECK(V.size(2) == grad_output.size(2), "V and grad_output head dimensions must match");

    const int batch = attention.size(0);
    const int N = attention.size(1);
    const int candidates = attention.size(2);
    const int head_dim = V.size(2);
    const int value_rows = V.size(1);
    auto grad_attention = at::empty_like(attention);
    auto grad_V_accum = at::zeros({batch, value_rows, head_dim}, V.options().dtype(torch::kFloat32));
    const dim3 grid(N, batch);
    constexpr int qk_threads = 256;
    constexpr int scatter_threads = 256;

    if (attention.scalar_type() == at::kFloat) {
        SMM_QmK_forward_v2_kernel<float><<<grid, qk_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.data_ptr<float>(), V.data_ptr<float>(), index.data_ptr<int>(), grad_attention.data_ptr<float>(),
            N, candidates, head_dim, value_rows, 0.0f);
        SMM_scatter_rows_backward_v2_kernel<float><<<grid, scatter_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            attention.data_ptr<float>(), grad_output.data_ptr<float>(), index.data_ptr<int>(), grad_V_accum.data_ptr<float>(),
            N, candidates, head_dim, value_rows);
    } else {
        TORCH_CHECK(attention.scalar_type() == at::kHalf, "optimized AmV backward supports float32 and float16 only");
        SMM_QmK_forward_v2_kernel<__half><<<grid, qk_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()), index.data_ptr<int>(),
            reinterpret_cast<__half*>(grad_attention.data_ptr<at::Half>()), N, candidates, head_dim, value_rows, 0.0f);
        SMM_scatter_rows_backward_v2_kernel<__half><<<grid, scatter_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(attention.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()), index.data_ptr<int>(),
            grad_V_accum.data_ptr<float>(), N, candidates, head_dim, value_rows);
    }
    auto grad_V = V.scalar_type() == at::kFloat ? grad_V_accum : grad_V_accum.to(V.scalar_type());
    return {grad_attention, grad_V};
}

at::Tensor SMM_QmK_forward_cuda(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &index) {
    return SMM_QmK_forward_cuda_v2(Q, K, index);
}

std::vector<at::Tensor> SMM_QmK_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &index) {
    return SMM_QmK_backward_cuda_v2(grad_output, Q, K, index);
}

at::Tensor SMM_AmV_forward_cuda(
    const at::Tensor &attention,
    const at::Tensor &V,
    const at::Tensor &index) {
    return SMM_AmV_forward_cuda_v2(attention, V, index);
}

std::vector<at::Tensor> SMM_AmV_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &attention,
    const at::Tensor &V,
    const at::Tensor &index) {
    return SMM_AmV_backward_cuda_v2(grad_output, attention, V, index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Keep stable Python API names while passing natural-layout K.
    m.def("SMM_QmK_forward_cuda", &SMM_QmK_forward_cuda,
          "Optimized sparse Q @ K forward (CUDA)");
    m.def("SMM_QmK_backward_cuda", &SMM_QmK_backward_cuda,
          "Optimized sparse Q @ K backward (CUDA)");
    m.def("SMM_AmV_forward_cuda", &SMM_AmV_forward_cuda,
          "Optimized sparse attention @ V forward (CUDA)");
    m.def("SMM_AmV_backward_cuda", &SMM_AmV_backward_cuda,
          "Optimized sparse attention @ V backward (CUDA)");

    // Explicit aliases are retained for side-by-side benchmarks.
    m.def("SMM_QmK_forward_cuda_v2", &SMM_QmK_forward_cuda_v2,
          "Optimized sparse Q @ K forward (CUDA)");
    m.def("SMM_QmK_backward_cuda_v2", &SMM_QmK_backward_cuda_v2,
          "Optimized sparse Q @ K backward (CUDA)");
    m.def("SMM_AmV_forward_cuda_v2", &SMM_AmV_forward_cuda_v2,
          "Optimized sparse attention @ V forward (CUDA)");
    m.def("SMM_AmV_backward_cuda_v2", &SMM_AmV_backward_cuda_v2,
          "Optimized sparse attention @ V backward (CUDA)");
}
