#pragma once

#include <torch/extension.h>
#include <vector>

// Stable Python function names with the v2 natural K layout [Batch, key_rows, head_dim].
at::Tensor SMM_QmK_forward_cuda(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &index);

std::vector<at::Tensor> SMM_QmK_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &index);

at::Tensor SMM_AmV_forward_cuda(
    const at::Tensor &attention,
    const at::Tensor &v,
    const at::Tensor &index);

std::vector<at::Tensor> SMM_AmV_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &attention,
    const at::Tensor &v,
    const at::Tensor &index);

// Explicit aliases for side-by-side benchmarks.
at::Tensor SMM_QmK_forward_cuda_v2(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &index);

std::vector<at::Tensor> SMM_QmK_backward_cuda_v2(
    const at::Tensor &grad_output,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &index);

at::Tensor SMM_AmV_forward_cuda_v2(
    const at::Tensor &attention,
    const at::Tensor &v,
    const at::Tensor &index);

std::vector<at::Tensor> SMM_AmV_backward_cuda_v2(
    const at::Tensor &grad_output,
    const at::Tensor &attention,
    const at::Tensor &v,
    const at::Tensor &index);
