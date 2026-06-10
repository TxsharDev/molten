"""
Molten: Fused GPU Kernel Generation from Mathematical Specifications

Build a DataflowGraph, compile to fused CUDA kernels.

    from molten import ZeroCompiler
    from molten.ir import DataflowGraph, TensorShape

    g = DataflowGraph("fused_rmsnorm")
    x = g.add_input("x", TensorShape([2048, 5120]))
    w = g.add_input("w", TensorShape([5120]))
    out = g.rms_norm(x, w, "norm")
    g.add_output(out)

    kernels = ZeroCompiler().compile(g)  # -> fused .cu files
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
