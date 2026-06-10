# Changelog

## v0.1.0 — 2026-06-10

First release. Math in, fused CUDA out.

### What's in it
- **Dataflow IR** — hardware-independent graph of ops (elementwise, reduction, matmul)
- **Fusion engine** — rule-based: elementwise chains, matmul+epilogue, reduction fusion
- **CUDA codegen** — generates complete .cu files with tiling, shared memory, vectorized access
- **JIT runtime** — `torch.utils.cpp_extension.load` pipeline, compile once, cache forever
- **@zero decorator** — trace a Python function, get a fused kernel

### Benchmarks (RTX 5090, Qwen3-30B dims)
- Molten-generated RMSNorm: **6.06x** over PyTorch eager at decode
- Hand-written fused RMSNorm+SiLU*Gate: **7.56x** over eager at decode
- Correctness: 19/19 configs PASS, max error ~1e-6 vs PyTorch

### Known gaps
- Elementwise codegen doesn't handle multi-input kernels via runtime yet (gelu_add, silu_gate)
- Large-sequence Molten kernels ~1.3x behind hand-written (needs vectorized loads)
- No half-precision (fp16/bf16) support yet
- No auto-tuning — block sizes are fixed at 256
