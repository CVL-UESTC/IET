import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

# RTX 4090 is sm_89.  Callers can override this for other GPUs, for example:
# TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install .
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

setup(
    name="smm_cuda",
    version="0.2.0",
    description="Optimized sparse QK/AV CUDA kernels with training backward",
    ext_modules=[
        CUDAExtension(
            name="smm_cuda",
            sources=[str(ROOT / "src" / "smm_cuda_v2.cu")],
            include_dirs=[str(ROOT / "src")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
