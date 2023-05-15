from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

workspace_dir = Path(__file__).parent

setup(
    name="fused_attn",
    ext_modules=[
        CUDAExtension(
            name="fused_attn",
            sources=[str(workspace_dir / "src" / "fused_attn_extention.cu")],
            include_dirs=[str(workspace_dir / "include")],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
