"""
Molten: Fused GPU Kernel Generation from Mathematical Specifications

Write math. Get fused CUDA kernels. No tile loops, no schedules,
no launch configs. The compiler handles everything.

    @zero
    def rmsnorm_rope_attn(x, w, freqs, k, v):
        x = x / rms(x) * w
        x = rotate(x, freqs)
        return softmax(x @ k.T / sqrt(d)) @ v

    # Emits a single fused CUDA kernel
"""

__version__ = "0.1.0"

from molten.decorator import zero, ZeroConfig
from molten.compiler import ZeroCompiler
from molten.ir import DataflowGraph, Op

__all__ = [
    "zero",
    "ZeroConfig",
    "ZeroCompiler",
    "DataflowGraph",
    "Op",
]
