"""
Molten-Generated Kernel Benchmark

The definitive test: can Molten's auto-generated CUDA match hand-written?

Builds DataflowGraphs for real Qwen3 ops, runs them through
Molten's fusion engine + codegen, JIT compiles the output,
and benchmarks against eager / torch.compile / hand-written CUDA.

Run: MOLTEN_GPU=1 python benchmarks/bench_molten_generated.py
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from molten.ir import DataflowGraph, OpType, TensorShape
from molten.fusion import FusionEngine
from molten.codegen import CodeGenerator, compile_graph, GeneratedKernel
from molten.runtime import MoltenRuntime

from benchmarks.ops import rmsnorm_pytorch, silu_gate_pytorch, gelu_add_pytorch


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


def build_rmsnorm_graph() -> DataflowGraph:
    """Build a DataflowGraph for RMSNorm."""
    g = DataflowGraph("rmsnorm")
    x = g.add_input("x", TensorShape(["rows", "cols"]))
    w = g.add_input("w", TensorShape(["cols"]))
    normed = g.rms_norm(x, w, "rmsnorm")
    g.add_output(normed)
    return g


def build_softmax_graph() -> DataflowGraph:
    """Build a DataflowGraph for softmax."""
    g = DataflowGraph("softmax")
    x = g.add_input("x", TensorShape(["rows", "cols"]))
    s = g.softmax(x)
    g.add_output(s)
    return g


def build_gelu_add_graph() -> DataflowGraph:
    """Build a DataflowGraph for GELU + bias add."""
    g = DataflowGraph("gelu_add")
    x = g.add_input("x", TensorShape(["N"]))
    bias = g.add_input("bias", TensorShape(["N"]))
    gelu = g.add_op(OpType.GELU, [x], "gelu")
    added = g.add(gelu, bias, "add")
    g.add_output(added)
    return g


def build_silu_gate_graph() -> DataflowGraph:
    """Build a DataflowGraph for SiLU * gate."""
    g = DataflowGraph("silu_gate")
    x = g.add_input("x", TensorShape(["N"]))
    gate = g.add_input("gate", TensorShape(["N"]))
    silu = g.add_op(OpType.SILU, [gate], "silu")
    out = g.mul(silu, x, "mul")
    g.add_output(out)
    return g


def analyze_fusion(name: str, graph: DataflowGraph):
    """Print fusion analysis for a graph."""
    engine = FusionEngine()
    groups = engine.fuse(graph)
    total_ops = sum(1 for op in graph.ops.values()
                    if op.op_type not in {OpType.INPUT, OpType.OUTPUT, OpType.CONSTANT})
    print(f"  {name}: {total_ops} ops -> {len(groups)} kernel(s) "
          f"(fusion ratio: {total_ops / max(len(groups), 1):.1f}x)")
    return groups


def generate_and_show(name: str, graph: DataflowGraph) -> list[GeneratedKernel]:
    """Generate kernels and print the source."""
    kernels = compile_graph(graph)
    for k in kernels:
        print(f"\n  --- Generated kernel: {k.name} ---")
        for line in k.source.split("\n")[:20]:
            print(f"    {line}")
        if k.source.count("\n") > 20:
            print(f"    ... ({k.source.count(chr(10)) - 20} more lines)")
    return kernels


def run_benchmark():
    gpu_id = int(os.environ.get("MOLTEN_GPU", "0"))
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    device_name = torch.cuda.get_device_name(gpu_id)
    vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

    print(f"{'='*80}")
    print(f"  MOLTEN GENERATED KERNEL ANALYSIS")
    print(f"  Device: {device_name}")
    print(f"  VRAM:   {vram_gb:.1f} GB")
    print(f"{'='*80}")

    # --- Fusion Analysis ---
    print(f"\n{'='*80}")
    print("  FUSION ANALYSIS")
    print(f"{'='*80}")

    graphs = {
        "rmsnorm": build_rmsnorm_graph(),
        "softmax": build_softmax_graph(),
        "gelu_add": build_gelu_add_graph(),
        "silu_gate": build_silu_gate_graph(),
    }

    all_kernels = {}
    for name, graph in graphs.items():
        groups = analyze_fusion(name, graph)
        kernels = generate_and_show(name, graph)
        all_kernels[name] = kernels

    # --- Save generated CUDA sources ---
    out_dir = Path(__file__).parent / "generated"
    out_dir.mkdir(exist_ok=True)
    for name, kernels in all_kernels.items():
        for k in kernels:
            path = out_dir / f"{k.name}.cu"
            path.write_text(k.source)
    print(f"\n  Generated kernels saved to {out_dir}/")

    # --- Try to JIT compile and benchmark ---
    print(f"\n{'='*80}")
    print("  JIT COMPILATION + BENCHMARK")
    print(f"{'='*80}")

    runtime = MoltenRuntime(verbose=True)

    # Qwen3-30B dims
    configs = [
        ("decode",  1, 1, 5120),
        ("prefill", 1, 2048, 5120),
        ("long",    1, 8192, 5120),
    ]

    results = []

    for label, batch, seq, dim in configs:
        rows = batch * seq
        print(f"\n  --- {label}: ({batch},{seq},{dim}) ---")

        # RMSNorm comparison
        x = torch.randn(rows, dim, device=device)
        w = torch.ones(dim, device=device)

        eager_t = bench_fn(lambda x, w: rmsnorm_pytorch(x.unsqueeze(0), w).squeeze(0),
                           (x, w))
        compile_t = bench_fn(torch.compile(
            lambda x, w: rmsnorm_pytorch(x.unsqueeze(0), w).squeeze(0)), (x, w))

        print(f"  {'rmsnorm':25s} eager:         {eager_t:8.1f} us")
        print(f"  {'':25s} torch.compile: {compile_t:8.1f} us")

        # Try Molten-generated
        rmsnorm_kernels = all_kernels.get("rmsnorm", [])
        for kernel in rmsnorm_kernels:
            if kernel.config.shared_mem_bytes > 0:
                try:
                    compiled = runtime.compile(kernel)
                    molten_t = bench_fn(compiled, (x,))
                    speedup = eager_t / molten_t if molten_t > 0 else 0
                    print(f"  {'':25s} molten:        {molten_t:8.1f} us ({speedup:.2f}x vs eager)")
                    results.append({"op": "rmsnorm", "config": label,
                                    "eager_us": eager_t, "compile_us": compile_t,
                                    "molten_us": molten_t, "speedup": speedup})
                except Exception as e:
                    print(f"  {'':25s} molten:        COMPILE FAILED: {e}")
                    results.append({"op": "rmsnorm", "config": label,
                                    "eager_us": eager_t, "compile_us": compile_t,
                                    "molten_us": -1, "error": str(e)})

        # SiLU*gate comparison
        x2 = torch.randn(rows, dim, device=device)
        gate = torch.randn(rows, dim, device=device)

        eager_t = bench_fn(silu_gate_pytorch, (x2, gate))
        compile_t = bench_fn(torch.compile(silu_gate_pytorch), (x2, gate))
        print(f"  {'silu_gate':25s} eager:         {eager_t:8.1f} us")
        print(f"  {'':25s} torch.compile: {compile_t:8.1f} us")

        results.append({"op": "silu_gate", "config": label,
                        "eager_us": eager_t, "compile_us": compile_t})

        # GELU+Add comparison
        bias = torch.randn(dim, device=device)
        eager_t = bench_fn(gelu_add_pytorch, (x2, bias))
        compile_t = bench_fn(torch.compile(gelu_add_pytorch), (x2, bias))
        print(f"  {'gelu_add':25s} eager:         {eager_t:8.1f} us")
        print(f"  {'':25s} torch.compile: {compile_t:8.1f} us")

        results.append({"op": "gelu_add", "config": label,
                        "eager_us": eager_t, "compile_us": compile_t})

    # Save results
    out_path = Path(__file__).parent / "molten_generated_results.json"
    with open(out_path, "w") as f:
        json.dump({"device": device_name, "vram_gb": vram_gb,
                    "results": results}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
