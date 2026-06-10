<p align="center">
  <h1 align="center">MOLTEN</h1>
  <p align="center"><i>Math melted into fused GPU kernels</i></p>
  <p align="center">Fused CUDA Kernel Generation from Mathematical Specifications</p>
  <p align="center">
    <a href="https://github.com/TxsharDev/molten">GitHub</a> · <a href="#citation">Paper</a> · <a href="#install">Install</a>
  </p>
</p>

---

> **Why "Molten"?** Molten metal is fluid, fused, white-hot — separate elements merged into one continuous pour. That's kernel fusion: separate operations melted together into a single GPU kernel, eliminating every memory round-trip between them. The math goes in fluid. The kernel comes out solid.

---

## The Problem

Every new model architecture needs custom CUDA. RMSNorm, RoPE, GQA, MoE routing — each requires hand-written, hand-fused kernels. Teams of 50+ engineers, weeks per kernel. Triton helps but still needs tile loops. TVM needs schedules. Nobody takes raw math and emits fused kernels.

## How Molten Works

Write math. Get fused CUDA.

```python
from molten import zero

@zero
def rmsnorm_rope_attn(x, w, freqs, k, v):
    x = x / rms(x) * w           # RMSNorm
    x = rotate(x, freqs)          # RoPE
    return softmax(x @ k.T) @ v   # Attention
```

Molten:
1. Traces into a dataflow graph
2. Discovers fusion opportunities across arbitrary boundaries
3. Generates CUDA with correct tiling, shared memory, vectorized access
4. Outputs `.cu` files

```
Python function
       │
   FX Tracer → DataflowGraph
       │
   Optimizer (constant fold, identity elim)
       │
   Fusion Engine (elementwise, matmul+epilog, reduction)
       │
   Code Generator → .cu files
```

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from molten import zero

@zero
def fused_gelu_add(x, bias):
    return gelu(x + bias)

# first call: trace + compile + cache
# subsequent: cached kernel
output = fused_gelu_add(x, bias)
```

## Programmatic API

```python
from molten import ZeroCompiler
from molten.ir import DataflowGraph, TensorShape

g = DataflowGraph("my_kernel")
x = g.add_input("x", TensorShape([4, 512]))
w = g.add_input("w", TensorShape([512]))
normed = g.rms_norm(x, w, "norm")
g.add_output(normed)

compiler = ZeroCompiler(verbose=True)
kernels = compiler.compile(g)
compiler.save(kernels, "output/")
```

## Fusion Rules

| Pattern | Result | Savings |
|---------|--------|---------|
| Elementwise → Elementwise | 1 kernel | -1 memory round-trip per op |
| MatMul → Bias → Activation | 1 kernel | -2 round-trips |
| RMSNorm (reduce + normalize) | 1 kernel | -1 intermediate buffer |
| Softmax (max + exp + sum + div) | 1 kernel | -3 round-trips |
| Chain of N elementwise ops | 1 kernel | -(N-1) round-trips |

## Citation

```bibtex
@article{sharma2025molten,
  title={Molten: Fused GPU Kernel Generation from Mathematical Specifications},
  author={Sharma, Tushar},
  year={2025},
  url={https://github.com/TxsharDev/molten}
}
```

## License

Apache-2.0 — Alia Labs
