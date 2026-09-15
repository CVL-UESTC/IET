# ops_smm (v2)

This is the active, speed-preserving SMM implementation in this IET checkout.
It installs the `smm_cuda` module and exports the four function names used by
`basicsr/archs/iet_arch.py`.  That model file already passes natural-layout K;
no additional patch or environment switch is required.

This directory contains the standard v2 QK/AV implementation and its training
backward.  The experimental Flash-style fused forward is not included.

## Interface

The extension module is named `smm_cuda` and exports:

- `SMM_QmK_forward_cuda(q, k, index) -> logits`
- `SMM_QmK_backward_cuda(grad_logits, q, k, index)`
- `SMM_AmV_forward_cuda(attention, v, index) -> output`
- `SMM_AmV_backward_cuda(grad_output, attention, v, index)`

QK uses `q: [B,N,D]` and natural-layout `k: [B,M,D]`.  AV uses
`v: [B,M,D]`; `index/attention` are `[B,N,K]`.  Inputs must be contiguous CUDA
tensors; floating inputs may be FP32 or FP16, and `index` must be int32.
Negative indices are padding: QK forward emits `-inf`, while all corresponding
backward gradients are zero.  Repeated indices are accumulated with FP32
atomics.

The module also exports `_v2` aliases for side-by-side benchmarks.

## Build and test

For an RTX 4090 (`sm_89`):

```bash
cd ops_smm
conda activate iet_py311_smm_v2
TORCH_CUDA_ARCH_LIST=8.9 ./make.sh
python test_correctness.py
python test_correctness.py --dtype float16
```

Set `TORCH_CUDA_ARCH_LIST` to the target compute capability when building for
another GPU.  After installation, use the IET training and testing commands
unchanged.

The implementation is covered by the repository-level `LICENSE.txt`.
