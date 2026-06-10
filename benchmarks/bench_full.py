"""
Full benchmark: PyTorch eager vs torch.compile vs hand-written CUDA.

This is the definitive benchmark for Molten. Shows:
1. How much perf is left on the table by eager PyTorch
2. How much torch.compile captures
3. How much a hand-fused kernel captures
4. The gap Molten needs to close

Run: python benchmarks/bench_full.py
"""

import torch
import time
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.ops import (
    rmsnorm_pytorch, silu_gate_pytorch, gelu_add_pytorch,
    fused_rmsnorm_silu_gate_pytorch,
    make_rmsnorm_inputs, make_silu_gate_inputs,
)


def bench_fn(fn, args, warmup=50, trials=200) -> float:
    """Returns median time in microseconds."""
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
        times.append(start.elapsed_time(end) * 1000)

    times.sort()
    q1 = len(times) // 4
    q3 = 3 * len(times) // 4
    return sum(times[q1:q3]) / (q3 - q1)


def load_reference_kernels():
    """JIT compile the hand-written CUDA reference kernels."""
    from torch.utils.cpp_extension import load
    kernel_path = Path(__file__).parent / "reference_kernels.cu"
    if not kernel_path.exists():
        print("reference_kernels.cu not found, skipping hand-written benchmarks")
        return None

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)

    print("Compiling reference CUDA kernels...")
    module = load(
        name="molten_reference",
        sources=[str(kernel_path)],
        build_directory=str(build_dir),
        verbose=False,
    )
    print("Reference kernels compiled.")
    return module


def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    gpu_id = int(os.environ.get("MOLTEN_GPU", "0"))
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    device_name = torch.cuda.get_device_name(gpu_id)
    vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

    print(f"{'='*80}")
    print(f"  MOLTEN KERNEL BENCHMARK")
    print(f"  Device: {device_name}")
    print(f"  VRAM:   {vram_gb:.1f} GB")
    print(f"{'='*80}")

    # Load hand-written kernels
    ref = load_reference_kernels()

    # Qwen3-30B dimensions: hidden=5120, heads=40, head_dim=128, ffn=25600
    configs = [
        {"name": "decode",        "batch": 1,  "seq": 1,    "dim": 5120},
        {"name": "short_prefill", "batch": 1,  "seq": 512,  "dim": 5120},
        {"name": "medium",        "batch": 1,  "seq": 2048, "dim": 5120},
        {"name": "long",          "batch": 1,  "seq": 8192, "dim": 5120},
        {"name": "batched",       "batch": 4,  "seq": 2048, "dim": 5120},
    ]

    all_results = []

    for cfg in configs:
        b, s, d = cfg["batch"], cfg["seq"], cfg["dim"]
        label = cfg["name"]
        print(f"\n{'-'*80}")
        print(f"  {label}: batch={b}, seq={s}, dim={d}")
        print(f"{'-'*80}")
        print(f"  {'op':30s} {'method':20s} {'time (us)':>10s} {'speedup':>8s}")
        print(f"  {'-'*70}")

        # --- RMSNorm ---
        x, w = make_rmsnorm_inputs(b, s, d, device)
        eager_t = bench_fn(rmsnorm_pytorch, (x, w))
        compile_t = bench_fn(torch.compile(rmsnorm_pytorch), (x, w))

        times = {"eager": eager_t, "torch.compile": compile_t}
        if ref:
            ref_t = bench_fn(ref.fused_rmsnorm, (x, w, 1e-6))
            times["hand-written CUDA"] = ref_t

        for method, t in times.items():
            sp = f"{eager_t/t:.2f}x" if t > 0 else "-"
            print(f"  {'rmsnorm':30s} {method:20s} {t:10.1f} {sp:>8s}")
            all_results.append({"op": "rmsnorm", "method": method,
                                "config": label, "time_us": t,
                                "speedup_vs_eager": eager_t/t if t > 0 else 0})

        # --- SiLU * gate ---
        x, gate = make_silu_gate_inputs(b, s, d, device)
        eager_t = bench_fn(silu_gate_pytorch, (x, gate))
        compile_t = bench_fn(torch.compile(silu_gate_pytorch), (x, gate))

        times = {"eager": eager_t, "torch.compile": compile_t}
        if ref:
            ref_t = bench_fn(ref.fused_silu_gate, (x, gate))
            times["hand-written CUDA"] = ref_t

        for method, t in times.items():
            sp = f"{eager_t/t:.2f}x" if t > 0 else "-"
            print(f"  {'silu_gate':30s} {method:20s} {t:10.1f} {sp:>8s}")
            all_results.append({"op": "silu_gate", "method": method,
                                "config": label, "time_us": t,
                                "speedup_vs_eager": eager_t/t if t > 0 else 0})

        # --- GELU + Add ---
        x = torch.randn(b, s, d, device=device)
        bias = torch.randn(d, device=device)
        eager_t = bench_fn(gelu_add_pytorch, (x, bias))
        compile_t = bench_fn(torch.compile(gelu_add_pytorch), (x, bias))

        times = {"eager": eager_t, "torch.compile": compile_t}
        if ref:
            ref_t = bench_fn(ref.fused_gelu_add, (x, bias))
            times["hand-written CUDA"] = ref_t

        for method, t in times.items():
            sp = f"{eager_t/t:.2f}x" if t > 0 else "-"
            print(f"  {'gelu_add':30s} {method:20s} {t:10.1f} {sp:>8s}")
            all_results.append({"op": "gelu_add", "method": method,
                                "config": label, "time_us": t,
                                "speedup_vs_eager": eager_t/t if t > 0 else 0})

        # --- Fused RMSNorm + SiLU*gate (the money shot) ---
        # Note: In real Qwen3 FFN, RMSNorm operates on hidden_dim (5120) and
        # SiLU*gate operates on ffn_intermediate (25600). This benchmark keeps
        # all tensors at hidden_dim to isolate the fusion benefit, not model the
        # full FFN pipeline (which includes two large matmuls between these ops).
        x, w = make_rmsnorm_inputs(b, s, d, device)
        gate = torch.randn_like(x)

        def separate(x, w, gate):
            return silu_gate_pytorch(rmsnorm_pytorch(x, w), gate)

        eager_t = bench_fn(separate, (x, w, gate))
        compile_t = bench_fn(torch.compile(fused_rmsnorm_silu_gate_pytorch), (x, w, gate))

        times = {"eager (3 ops)": eager_t, "torch.compile": compile_t}
        if ref:
            ref_t = bench_fn(ref.fused_rmsnorm_silu_gate, (x, w, gate, 1e-6))
            times["hand-written CUDA"] = ref_t

        for method, t in times.items():
            sp = f"{eager_t/t:.2f}x" if t > 0 else "-"
            print(f"  {'rmsnorm+silu_gate (FUSED)':30s} {method:20s} {t:10.1f} {sp:>8s}")
            all_results.append({"op": "rmsnorm+silu_gate", "method": method,
                                "config": label, "time_us": t,
                                "speedup_vs_eager": eager_t/t if t > 0 else 0})

    # Save
    out_path = Path(__file__).parent / "bench_results.json"
    with open(out_path, "w") as f:
        json.dump({"device": device_name, "vram_gb": vram_gb,
                    "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
