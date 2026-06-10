"""
Molten Kernel Benchmark Suite

Benchmarks Molten-generated fused kernels against:
1. PyTorch eager (baseline)
2. torch.compile (inductor)
3. Hand-written CUDA reference kernels

All ops based on Qwen3/Llama transformer architecture.
Run with: python benchmarks/bench_kernels.py
"""

import torch
import torch.nn.functional as F
import time
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.ops import (
    rmsnorm_pytorch, silu_gate_pytorch, softmax_pytorch,
    gelu_add_pytorch, fused_rmsnorm_silu_gate_pytorch,
    make_rmsnorm_inputs, make_silu_gate_inputs,
)


@dataclass
class BenchResult:
    op_name: str
    method: str
    shape: str
    dtype: str
    time_us: float
    bandwidth_gb_s: float = 0.0
    tflops: float = 0.0

    def __str__(self):
        return (f"{self.op_name:30s} | {self.method:15s} | {self.shape:20s} | "
                f"{self.time_us:8.1f} us | {self.bandwidth_gb_s:6.1f} GB/s")


def bench_fn(fn, args, warmup=50, trials=200) -> float:
    """Benchmark a function, return median time in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    # Median of middle 50%
    q1 = len(times) // 4
    q3 = 3 * len(times) // 4
    return sum(times[q1:q3]) / (q3 - q1)


def memory_bandwidth(nbytes: int, time_us: float) -> float:
    """Compute effective memory bandwidth in GB/s."""
    if time_us == 0:
        return 0.0
    return nbytes / (time_us * 1e-6) / 1e9


def bench_rmsnorm(batch: int, seq: int, dim: int, device: str = "cuda"):
    """Benchmark RMSNorm: PyTorch eager vs torch.compile."""
    results = []
    shape_str = f"({batch},{seq},{dim})"
    x, w = make_rmsnorm_inputs(batch, seq, dim, device)
    nbytes = x.numel() * x.element_size() * 3  # read x, read w, write out

    # PyTorch eager
    t = bench_fn(rmsnorm_pytorch, (x, w))
    results.append(BenchResult("rmsnorm", "eager", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    # torch.compile
    compiled = torch.compile(rmsnorm_pytorch)
    t = bench_fn(compiled, (x, w))
    results.append(BenchResult("rmsnorm", "torch.compile", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    return results


def bench_silu_gate(batch: int, seq: int, dim: int, device: str = "cuda"):
    """Benchmark SiLU * gate."""
    results = []
    shape_str = f"({batch},{seq},{dim})"
    x, gate = make_silu_gate_inputs(batch, seq, dim, device)
    nbytes = x.numel() * x.element_size() * 3

    t = bench_fn(silu_gate_pytorch, (x, gate))
    results.append(BenchResult("silu_gate", "eager", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    compiled = torch.compile(silu_gate_pytorch)
    t = bench_fn(compiled, (x, gate))
    results.append(BenchResult("silu_gate", "torch.compile", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    return results


def bench_softmax(batch: int, seq: int, dim: int, device: str = "cuda"):
    """Benchmark softmax."""
    results = []
    shape_str = f"({batch},{seq},{dim})"
    x = torch.randn(batch * seq, dim, device=device)
    nbytes = x.numel() * x.element_size() * 2

    t = bench_fn(F.softmax, (x, -1))
    results.append(BenchResult("softmax", "eager", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    compiled_softmax = torch.compile(lambda x: F.softmax(x, dim=-1))
    t = bench_fn(compiled_softmax, (x,))
    results.append(BenchResult("softmax", "torch.compile", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    return results


def bench_fused_rmsnorm_silu_gate(batch: int, seq: int, dim: int,
                                   device: str = "cuda"):
    """
    The money benchmark: fused RMSNorm + SiLU gate.
    Three separate ops that should be one kernel.
    """
    results = []
    shape_str = f"({batch},{seq},{dim})"
    x, w = make_rmsnorm_inputs(batch, seq, dim, device)
    gate = torch.randn_like(x)
    nbytes = x.numel() * x.element_size() * 4  # read x, w, gate; write out

    # Separate ops (eager)
    def separate(x, w, gate):
        normed = rmsnorm_pytorch(x, w)
        return silu_gate_pytorch(normed, gate)

    t = bench_fn(separate, (x, w, gate))
    results.append(BenchResult("rmsnorm+silu_gate", "eager (separate)", shape_str,
                               str(x.dtype), t, memory_bandwidth(nbytes, t)))

    # torch.compile (should auto-fuse)
    compiled = torch.compile(fused_rmsnorm_silu_gate_pytorch)
    t = bench_fn(compiled, (x, w, gate))
    results.append(BenchResult("rmsnorm+silu_gate", "torch.compile", shape_str,
                               str(x.dtype), t, memory_bandwidth(nbytes, t)))

    return results


def bench_gelu_add(batch: int, seq: int, dim: int, device: str = "cuda"):
    """Benchmark GELU + add (common epilogue)."""
    results = []
    shape_str = f"({batch},{seq},{dim})"
    x = torch.randn(batch, seq, dim, device=device)
    bias = torch.randn(dim, device=device)
    nbytes = x.numel() * x.element_size() * 3

    t = bench_fn(gelu_add_pytorch, (x, bias))
    results.append(BenchResult("gelu_add", "eager", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    compiled = torch.compile(gelu_add_pytorch)
    t = bench_fn(compiled, (x, bias))
    results.append(BenchResult("gelu_add", "torch.compile", shape_str, str(x.dtype),
                               t, memory_bandwidth(nbytes, t)))

    return results


def run_all_benchmarks():
    """Run all benchmarks at Qwen3-30B scale dimensions."""
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("=" * 95)

    # Qwen3-30B dimensions
    # hidden_size=5120, num_heads=40, head_dim=128, intermediate_size=25600
    configs = [
        # (batch, seq, dim) — representative workloads
        (1, 1, 5120),       # decode: single token
        (1, 512, 5120),     # short prefill
        (1, 2048, 5120),    # medium prefill
        (1, 8192, 5120),    # long prefill
        (4, 2048, 5120),    # batched prefill
    ]

    all_results = []

    for batch, seq, dim in configs:
        print(f"\n--- batch={batch}, seq={seq}, dim={dim} ---")

        for bench_fn_factory in [bench_rmsnorm, bench_silu_gate, bench_softmax,
                                  bench_gelu_add, bench_fused_rmsnorm_silu_gate]:
            try:
                results = bench_fn_factory(batch, seq, dim)
                for r in results:
                    print(r)
                    all_results.append(asdict(r))
            except Exception as e:
                print(f"  SKIP {bench_fn_factory.__name__}: {e}")

    # Save results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": device_name,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 95)
    print("SUMMARY: speedup of torch.compile over eager")
    print("=" * 95)
    eager_times = {}
    compile_times = {}
    for r in all_results:
        key = (r["op_name"], r["shape"])
        if "eager" in r["method"]:
            eager_times[key] = r["time_us"]
        elif "compile" in r["method"]:
            compile_times[key] = r["time_us"]

    for key in eager_times:
        if key in compile_times:
            speedup = eager_times[key] / compile_times[key]
            print(f"  {key[0]:30s} {key[1]:20s}  {speedup:.2f}x")


if __name__ == "__main__":
    run_all_benchmarks()
