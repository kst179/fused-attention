from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

workspace_dir = Path(__file__).parent
cuda_fused_attention_dir = workspace_dir / "cuda_fused_attention"
extention_file = cuda_fused_attention_dir / "fused_attn_extention.cu"

setup(
    name="fused_attention",
    ext_modules=[
        CUDAExtension(
            name="fused_attention",
            sources=[extention_file.as_posix()],
            include_dirs=[cuda_fused_attention_dir.as_posix()],
            extra_compile_args=["-std=c++17", "-O3"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
