#!/usr/bin/env python3
"""Compare SMM v2 forward/backward with a PyTorch reference."""

import argparse
import json

import torch
from torch.autograd import Function

import smm_cuda


class SparseQK(Function):
    @staticmethod
    def forward(ctx, q, k, index):
        ctx.save_for_backward(q, k, index)
        return smm_cuda.SMM_QmK_forward_cuda(
            q.contiguous(), k.contiguous(), index.contiguous()
        )

    @staticmethod
    def backward(ctx, grad_output):
        q, k, index = ctx.saved_tensors
        grad_q, grad_k = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(),
            q.contiguous(),
            k.contiguous(),
            index.contiguous(),
        )
        return grad_q, grad_k, None


class SparseAV(Function):
    @staticmethod
    def forward(ctx, attention, v, index):
        ctx.save_for_backward(attention, v, index)
        return smm_cuda.SMM_AmV_forward_cuda(
            attention.contiguous(), v.contiguous(), index.contiguous()
        )

    @staticmethod
    def backward(ctx, grad_output):
        attention, v, index = ctx.saved_tensors
        grad_attention, grad_v = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(),
            attention.contiguous(),
            v.contiguous(),
            index.contiguous(),
        )
        return grad_attention, grad_v, None


def gather_rows(matrix, index):
    safe_index = index.clamp_min(0).long()
    return torch.gather(
        matrix.unsqueeze(1).expand(-1, index.shape[1], -1, -1),
        2,
        safe_index.unsqueeze(-1).expand(-1, -1, -1, matrix.shape[-1]),
    )


def reference_qk(q, k, index):
    logits = (q.unsqueeze(2) * gather_rows(k, index)).sum(dim=-1)
    return logits.masked_fill(index < 0, -torch.inf)


def reference_av(attention, v, index):
    valid_attention = attention.masked_fill(index < 0, 0)
    return (valid_attention.unsqueeze(-1) * gather_rows(v, index)).sum(dim=2)


def max_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def run(dtype):
    torch.manual_seed(2718)
    batch, query_rows, value_rows, head_dim, candidates = 2, 37, 43, 40, 31
    q0 = torch.randn(batch, query_rows, head_dim, device="cuda", dtype=dtype)
    k0 = torch.randn(batch, value_rows, head_dim, device="cuda", dtype=dtype)
    v0 = torch.randn_like(k0)
    index = torch.randint(
        0,
        value_rows,
        (batch, query_rows, candidates),
        device="cuda",
        dtype=torch.int32,
    )
    index[:, :, 1] = index[:, :, 0]  # repeated scatter destination
    index[:, :, -1] = -1             # IET boundary padding
    upstream = torch.randn_like(q0)

    def attention_result(qk, av):
        q = q0.detach().clone().requires_grad_(True)
        k = k0.detach().clone().requires_grad_(True)
        v = v0.detach().clone().requires_grad_(True)
        logits = qk(q, k, index)
        attention = torch.softmax(logits, dim=-1)
        output = av(attention, v, index)
        gradients = torch.autograd.grad((output * upstream).sum(), (q, k, v))
        return output, logits, gradients

    reference = attention_result(reference_qk, reference_av)
    actual = attention_result(SparseQK.apply, SparseAV.apply)

    # Test grad_attention directly: invalid entries must be zero, not -inf.
    attention = torch.randn(
        batch, query_rows, candidates, device="cuda", dtype=dtype, requires_grad=True
    )
    actual_av = SparseAV.apply(attention, v0, index)
    actual_grad_attention = torch.autograd.grad(
        (actual_av * upstream).sum(), attention
    )[0]
    invalid_gradient = actual_grad_attention[index < 0]

    valid = index >= 0
    checks = {
        "output_max_abs_error": max_error(actual[0], reference[0]),
        "logits_max_abs_error": max_error(actual[1][valid], reference[1][valid]),
        "invalid_logits_are_neg_inf": bool(torch.isneginf(actual[1][~valid]).all()),
        "grad_q_max_abs_error": max_error(actual[2][0], reference[2][0]),
        "grad_k_max_abs_error": max_error(actual[2][1], reference[2][1]),
        "grad_v_max_abs_error": max_error(actual[2][2], reference[2][2]),
        "invalid_grad_attention_max_abs": invalid_gradient.float().abs().max().item(),
    }
    # FP16 reductions accumulate in FP32 inside the CUDA kernels and are then
    # rounded once on output.  A 0.04 absolute bound covers one observed FP16
    # quantization step at the tested gradient magnitude.
    tolerance = 3.0e-5 if dtype == torch.float32 else 4.0e-2
    error_fields = [key for key in checks if "error" in key or key.endswith("max_abs")]
    if not checks["invalid_logits_are_neg_inf"]:
        raise AssertionError(checks)
    if any(checks[key] > tolerance for key in error_fields):
        raise AssertionError(checks)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    checks = run(dtype)
    torch.cuda.synchronize()
    print(json.dumps({"device": torch.cuda.get_device_name(), "dtype": args.dtype, **checks}, indent=2))


if __name__ == "__main__":
    main()
