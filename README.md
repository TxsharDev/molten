<p align="center">
  <h1 align="center">MOLTEN</h1>
  <p align="center"><b>Write the math. Get the kernel.</b></p>
  <p align="center">
    <a href="https://pypi.org/project/alia-molten/"><img src="https://img.shields.io/pypi/v/alia-molten?color=blue&label=PyPI" alt="PyPI"></a>
    <a href="https://github.com/TxsharDev/molten/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
    <a href="#benchmarks"><img src="https://img.shields.io/badge/RTX%205090-4.6x%20vs%20torch.compile-red" alt="Speedup"></a>
  </p>
</p>

---

Molten turns mathematical operation specs into fused, portable CUDA kernels.

No tile loops. No schedules. No framework lock-in. The output is a `.cu` file. It compiles with `nvcc`. It runs without PyTorch.

Built by [Tushar Sharma](https://github.com/TxsharDev) at Alia Labs.

## Install

```bash
pip install alia-molten
```

## 30 Seconds to a Fused Kernel

```python
from molten import ZeroCompiler
from molten.ir import DataflowGraph, TensorShape

g = DataflowGraph("fused_rmsnorm")
x = g.add_input("x", TensorShape([2048, 5120]))
w = g.add_input("w", TensorShape([5120]))
out = g.rms_norm(x, w, "norm")
g.add_output(out)

compiler = ZeroCompiler()
kernels = compiler.compile(g)        # 3 ops -> 1 kernel
compiler.save(kernels, "output/")    # standalone .cu file
```

That's it. Three operations. One kernel. Zero CUDA written by hand.

## What Happens Under the Hood

```
Math Spec -> DataflowGraph -> Optimizer -> Fusion Engine -> CUDA Codegen -> .cu
```

The fusion engine knows six rules:

| Pattern | What It Does |
|---------|-------------|
| Elementwise chain | Fuses N ops into 1. Kills N-1 memory round-trips. |
| MatMul + bias + activation | Epilogue fusion. One kernel does matmul, adds bias, applies GELU. |
| RMSNorm | Fuses reduce + normalize + scale. One pass over the data. |
| Softmax | Fuses max + exp + sum + divide. Three passes become one. |

## Benchmarks

RTX 5090. Same session. 19/19 correctness. Also validated on RTX 4090 and H100 SXM.

**Molten-generated RMSNorm (zero hand-written CUDA):**

| | Eager | torch.compile | Molten | vs Compile |
|--|-------|---------------|--------|------------|
| **decode** (1 token) | 167 us | 127 us | **28 us** | **4.6x** |
| **prefill** (2048 tokens) | 160 us | 96 us | **55 us** | **1.7x** |
| **long** (8192 tokens) | 793 us | 323 us | 394 us | 0.82x |

Molten wins at decode and prefill. torch.compile wins at long sequences (scalar loads vs vectorized). That gap closes in v0.2.

**Hand-written fused RMSNorm+SiLU*gate (the target Molten is closing in on):**

| | Eager (3 ops) | Fused (1 kernel) | Speedup |
|--|--------------|-----------------|---------|
| **decode** | 207 us | **27 us** | **7.6x** |
| **prefill** | 347 us | **97 us** | **3.6x** |
| **long** | 1327 us | **403 us** | **3.3x** |

## Why Not torch.compile?

torch.compile generates Triton code tied to PyTorch. You can't deploy it without the full Python + PyTorch + Triton stack.

Molten generates a `.cu` file. Ship it to TensorRT, ONNX Runtime, a C++ server, a Jetson, whatever. It's just CUDA.

## Tested On

RTX 4090 | RTX 5090 | H100 SXM 80GB

Real model validation on Qwen2.5-7B (57 RMSNorm layers, hidden=3584).

## Citation

```bibtex
@article{sharma2026molten,
  title={Molten: Fused GPU Kernel Generation from Mathematical Specifications},
  author={Sharma, Tushar},
  year={2026},
  url={https://github.com/TxsharDev/molten}
}
```

## Roadmap

**v0.1 (current)** - IR, fusion engine, CUDA codegen, JIT runtime. RMSNorm and elementwise fusion proven. Scalar memory access.

**v0.2** - Vectorized loads (`float4`/`half2`). This closes the gap where torch.compile currently wins at long sequences. fp16 I/O benchmarked end-to-end. `@zero` decorator dispatches generated kernels directly.

**v0.3** - Attention fusion (Q@K softmax @V as one kernel). RoPE integration. Polyhedral loop optimization for complex fusion patterns. Auto-tuning via hardware counter feedback.

## License

Apache-2.0 | Alia Labs
